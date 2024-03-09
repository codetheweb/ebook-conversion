use crate::{
    huffman::HuffmanTable,
    tree::{HuffmanNode, HuffmanTree},
};

use bitvec::prelude::*;
use memchr::{memchr_iter, memmem};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
};
use thiserror::Error;

const MAX_DICTIONARY_ENTRY_LENGTH: usize = 2;
const NULL_PADDING_SYMBOL: [u8; 4] = [0, 0, 0, 0];

#[derive(Error, Debug)]
pub enum HuffmanEncodingError {
    // todo: shouldn't need this error
    #[error("Not enough unique data")]
    NotEnoughUniqueData,
}

pub struct HuffmanEncoder {
    table: HuffmanTable,
    code_length_to_max_code: HashMap<usize, u32>,
    byte_to_code: HashMap<Vec<u8>, BitVec<u8>>,
    compressed: BitVec<u8, Msb0>,
}

impl Default for HuffmanEncoder {
    fn default() -> Self {
        Self {
            table: HuffmanTable::default(),
            code_length_to_max_code: HashMap::new(),
            byte_to_code: HashMap::new(),
            compressed: BitVec::new(),
        }
    }
}

impl HuffmanEncoder {
    pub fn pack(&mut self, data: &[u8]) -> Result<(), HuffmanEncodingError> {
        println!("packing data: {:?}", data);

        let mut buffer: Vec<u8> = vec![data[0]];
        let mut entry_to_indices: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();

        for i in 0..data.len() {
            if i < data.len() - 1 {
                let next_byte = data[i + 1];
                let mut next_buffer = buffer.clone();
                next_buffer.push(next_byte);

                if entry_to_indices.contains_key(&next_buffer) {
                    buffer.push(next_byte);
                } else {
                    let starts_at = i.saturating_sub(buffer.len());
                    entry_to_indices
                        .entry(next_buffer)
                        .and_modify(|x| x.push(starts_at))
                        .or_insert(vec![starts_at]);

                    buffer.clear();
                }
            }
        }

        if buffer.len() > 0 {
            let starts_at = data.len().saturating_sub(buffer.len());
            entry_to_indices
                .entry(buffer)
                .and_modify(|x| x.push(starts_at))
                .or_insert(vec![starts_at]);
        }

        if entry_to_indices.len() > 256 {
            // todo: should trim
            panic!("Dictionary full");
        }

        let mut weights = HashMap::new();
        for (entry, indices) in entry_to_indices.iter() {
            weights.insert(entry, indices.len());
        }

        // todo: make sure this doesn't collide
        let null_padding_symbol = NULL_PADDING_SYMBOL.to_vec();
        weights.insert(&null_padding_symbol, 1);

        let mut tree = HuffmanTree::from_iter(weights.into_iter());
        tree.enforce_minimum_code_length(4);
        let codebook = tree.get_canonical_codebook();

        println!("tree {:?}", codebook);

        let mut code_length_to_max_code: HashMap<usize, u32> = HashMap::new();
        for code in codebook.values() {
            let code_length = code.len();
            // let max_code = (1 << code_length) - 1;
            let shifted = code.load::<u32>() << (32 - code_length);
            let shifted = shifted | u32::MAX >> (code_length);
            code_length_to_max_code
                .entry(code_length)
                .and_modify(|x| {
                    *x = (*x).max(shifted);
                })
                .or_insert(shifted);
        }

        println!("max codes: {:?}", code_length_to_max_code);

        // Construct code_dict
        for (symbol, code) in codebook.iter() {
            let code_length = code.len();
            let inverted_code = (code_length_to_max_code.get(&code_length).unwrap()
                >> (32 - code_length))
                - code.load::<u8>() as u32;
            let min_code = inverted_code << (8 - code_length);
            let max_code = inverted_code << (8 - code_length) | (1 << (8 - code_length)) - 1;

            for code in min_code..(max_code + 1) {
                if self.table.code_dict[code as usize].1 {
                    panic!("Code collision");
                }

                self.table.code_dict[code as usize] = (
                    code_length as u8,
                    true,
                    *code_length_to_max_code.get(&code_length).unwrap(),
                );
            }

            println!(
                "code: {:08b} ({}), min_code: {:08b}, max_code: {:08b}, symbol: {:?}",
                code.load::<u8>(),
                code_length,
                min_code,
                max_code,
                symbol
            );
        }

        let mut indicies_to_symbol = BTreeMap::new();

        for (symbol, indices) in entry_to_indices.iter() {
            for index in indices {
                indicies_to_symbol.insert(*index, symbol);
            }
        }

        println!("indicies to symbol: {:?}", indicies_to_symbol);

        for symbol in indicies_to_symbol.values() {
            self.write_symbol(*symbol, &codebook, &code_length_to_max_code)
        }

        // Padding
        while self.compressed.len() % 32 != 0 {
            self.write_symbol(
                &NULL_PADDING_SYMBOL.to_vec(),
                &codebook,
                &code_length_to_max_code,
            );
        }

        for (i, entry) in self.table.dictionary.iter().enumerate() {
            if let Some(entry) = entry {
                if entry.1 {
                    println!("dictionary {}: {:?}", i, entry);
                }
            }
        }

        Ok(())
    }

    fn write_symbol(
        &mut self,
        symbol: &Vec<u8>,
        codebook: &HashMap<&Vec<u8>, BitVec>,
        code_length_to_max_code: &HashMap<usize, u32>,
    ) {
        let code = codebook.get(symbol).unwrap();
        let dictionary_index = code.load::<u8>();

        println!("using dictionary index: {} {:?}", dictionary_index, symbol);

        if (*symbol == NULL_PADDING_SYMBOL) {
            self.table.dictionary[dictionary_index as usize] = Some((vec![], true));
        } else {
            self.table.dictionary[dictionary_index as usize] = Some(((*symbol).clone(), true));
        }

        let shifted_code = (dictionary_index as u32) << (32 - code.len());
        let partial_code: u32 = code_length_to_max_code.get(&code.len()).unwrap() - shifted_code;
        let partial_code: BitVec<_, Msb0> = BitVec::from_element(partial_code);

        println!(
            "writing {} for {:?}",
            partial_code[0..code.len()].to_bitvec(),
            symbol
        );

        self.compressed
            .append(&mut partial_code[0..code.len()].to_bitvec());
    }

    pub fn finish(self) -> (HuffmanTable, Vec<u8>) {
        (self.table, self.compressed.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use crate::huffman_decoder::HuffmanDecoder;

    use super::*;

    #[test]
    fn test_huffman_encoder() {
        let mut encoder = HuffmanEncoder {
            code_length_to_max_code: HashMap::new(),
            table: HuffmanTable::default(),
            compressed: BitVec::new(),
            byte_to_code: HashMap::new(),
        };
        let data = b"heyyyuuyyyu";
        encoder.pack(data).unwrap();
        let (table, packed) = encoder.finish();

        let mut decoder = HuffmanDecoder { table };

        let unpacked = decoder.unpack(&packed);
        println!(
            "unpacked test: {} {:?} {:?}",
            String::from_utf8_lossy(&unpacked),
            unpacked,
            data
        );
        assert_eq!(unpacked, data);
    }
}
