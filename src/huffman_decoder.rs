use super::huffman::HuffmanTable;
use byteorder::{BigEndian, ReadBytesExt};
use std::{collections::HashMap, io::Cursor};

pub struct HuffmanDecoder {
    pub table: HuffmanTable,
}

impl Default for HuffmanDecoder {
    fn default() -> Self {
        Self {
            table: HuffmanTable::default(),
        }
    }
}

impl HuffmanDecoder {
    pub fn unpack(&mut self, input: &[u8]) -> Vec<u8> {
        // Need len.
        let mut bits_left = input.len() * 8;
        let mut data = input.to_vec();
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

        println!("unpacking data: {:?}", data);

        let mut r = Cursor::new(data.as_slice());

        // X is a sliding window of 64 bits from data.
        let mut x = r.read_u64::<BigEndian>().unwrap();
        r.set_position(r.position() - 4);
        // -32 < n <= 32
        let mut n = 32i8;
        let mut unpacked = vec![];

        let mut last_code_len = 0;

        let mut min_and_max_indices_by_code_length: HashMap<u8, (u8, u8)> = HashMap::new();

        loop {
            if n <= 0 {
                x = r.read_u64::<BigEndian>().unwrap();
                r.set_position(r.position() - 4);
                n += 32;
            }

            println!("x:    {:064b} {}", x, n);

            let code = (x >> n) & ((1 << 32) - 1);
            println!("code: {:064b}", code);
            // println!(
            //     "code: {:b} shifted: {} shifted bit: {:08b} code dict: {:?}",
            //     code,
            //     code >> 24,
            //     code >> 24,
            //     self.table.code_dict[(code >> 24) as usize]
            // );
            println!("shifted code {:08b}", code >> 24);
            let (mut code_len, term, mut max_code) = self.table.code_dict[(code >> 24) as usize];

            // println!("term: {:?} {:?}", term, max_code);
            //
            // 7: min 26, max 52
            // 4: min 0, max

            if !term {
                while code < self.table.min_codes[code_len as usize] as u64 {
                    code_len += 1;
                }
                max_code = self.table.max_codes[code_len as usize];
            }

            // println!("code len: {:?}, bits remaining: {:?}", code_len, bits_left);

            n -= code_len as i8;
            bits_left = match bits_left.checked_sub(code_len as usize) {
                Some(bits_left) => bits_left,
                None => break,
            };

            // println!(
            //     "max_code: {:?}, code: {:?}, difference: {:b}, code_len: {:?}",
            //     max_code,
            //     code,
            //     max_code - code as u32,
            //     code_len
            // );
            let index = ((max_code as u64 - code) >> (32 - code_len)) as usize;

            println!(
                "code len: {:?}, max_code: {:032b}, index: {}",
                code_len, max_code, index
            );
            last_code_len = code_len;

            if code_len > 0 {
                // Make sure there's no overlap
                min_and_max_indices_by_code_length
                    .iter()
                    .for_each(|(len, (min, max))| {
                        if *len != code_len {
                            if *min <= index as u8 && index as u8 <= *max {
                                panic!("Overlap detected: {} <= {} <= {}", min, index, max);
                            }
                        }
                    });

                let (min_index, max_index) = min_and_max_indices_by_code_length
                    .entry(code_len)
                    .or_insert((index as u8, index as u8));
                *min_index = (*min_index).min(index as u8);
                *max_index = (*max_index).max(index as u8);
            }

            // println!("index: {:?}", index);
            let (mut slice, flag) = self.table.dictionary[index].clone().unwrap();
            if !flag {
                self.table.dictionary[index] = None;
                slice = self.unpack(&slice);
                self.table.dictionary[index] = Some((slice.clone(), true));
            }
            unpacked.extend_from_slice(&slice);
            println!("currently unpacked: {:?}", unpacked);
        }

        println!("unpacked: {:?}", unpacked);

        let mut printable_table = min_and_max_indices_by_code_length
            .iter()
            .collect::<Vec<_>>();
        printable_table.sort_by_key(|(code_len, _)| *code_len);

        for (code_len, (min_index, max_index)) in printable_table {
            println!(
                "code len: {:02}, min index: {:03}, max index: {:03}",
                code_len, min_index, max_index
            );
        }

        unpacked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huff() {
        let dictionary = vec![
            (hex::decode("20").unwrap(), true),
            (hex::decode("65").unwrap(), true),
            (hex::decode("74").unwrap(), true),
            (hex::decode("69").unwrap(), true),
            (hex::decode("61").unwrap(), true),
            (hex::decode("7573").unwrap(), true),
            (hex::decode("6f72").unwrap(), true),
            (hex::decode("2e20").unwrap(), true),
            (hex::decode("6973").unwrap(), true),
            (hex::decode("72").unwrap(), true),
            (hex::decode("75").unwrap(), true),
            (hex::decode("2c20").unwrap(), true),
            (hex::decode("6e").unwrap(), true),
            (hex::decode("6572").unwrap(), true),
            (hex::decode("2061").unwrap(), true),
            (hex::decode("656c").unwrap(), true),
            (hex::decode("6d").unwrap(), true),
            (hex::decode("6c").unwrap(), true),
            (hex::decode("70").unwrap(), true),
            (hex::decode("6f").unwrap(), true),
            (hex::decode("206e").unwrap(), true),
            (hex::decode("206c").unwrap(), true),
            (hex::decode("73").unwrap(), true),
            (hex::decode("63").unwrap(), true),
            (hex::decode("2065").unwrap(), true),
            (hex::decode("68").unwrap(), true),
            (hex::decode("66").unwrap(), true),
            (hex::decode("2c").unwrap(), true),
            (hex::decode("6974").unwrap(), true),
            (hex::decode("2076").unwrap(), true),
            (hex::decode("6c61").unwrap(), true),
            (hex::decode("756d").unwrap(), true),
            (hex::decode("696e").unwrap(), true),
            (hex::decode("6574").unwrap(), true),
            (hex::decode("7572").unwrap(), true),
            (hex::decode("2073").unwrap(), true),
            (hex::decode("206d").unwrap(), true),
            (hex::decode("616d").unwrap(), true),
            (hex::decode("6174").unwrap(), true),
            (hex::decode("67").unwrap(), true),
            (hex::decode("6573").unwrap(), true),
            (hex::decode("76").unwrap(), true),
            (hex::decode("6469").unwrap(), true),
            (hex::decode("756c").unwrap(), true),
            (hex::decode("6564").unwrap(), true),
            (hex::decode("656e").unwrap(), true),
            (hex::decode("7175").unwrap(), true),
            (hex::decode("62").unwrap(), true),
            (hex::decode("4e").unwrap(), true),
            (hex::decode("6f6e").unwrap(), true),
            (hex::decode("6173").unwrap(), true),
            (hex::decode("7574").unwrap(), true),
            (hex::decode("6962").unwrap(), true),
            (hex::decode("64").unwrap(), true),
            (hex::decode("7420").unwrap(), true),
            (hex::decode("2070").unwrap(), true),
            (hex::decode("6e61").unwrap(), true),
            (hex::decode("6163").unwrap(), true),
            (hex::decode("3e").unwrap(), true),
            (hex::decode("6374").unwrap(), true),
            (hex::decode("6775").unwrap(), true),
            (hex::decode("20636f").unwrap(), true),
            (hex::decode("7465").unwrap(), true),
            (hex::decode("6964").unwrap(), true),
            (hex::decode("2073697420616d6574").unwrap(), true),
            (hex::decode("7269").unwrap(), true),
            (hex::decode("7361").unwrap(), true),
            (hex::decode("3d22").unwrap(), true),
            (hex::decode("3c").unwrap(), true),
            (hex::decode("6c69").unwrap(), true),
            (hex::decode("22").unwrap(), true),
            (hex::decode("6f6c").unwrap(), true),
            (hex::decode("6563").unwrap(), true),
            (hex::decode("2064").unwrap(), true),
            (hex::decode("43").unwrap(), true),
            (hex::decode("6d70").unwrap(), true),
            (hex::decode("6f73").unwrap(), true),
            (hex::decode("6167").unwrap(), true),
            (hex::decode("6f64").unwrap(), true),
            (hex::decode("20766974616520").unwrap(), true),
            (hex::decode("78").unwrap(), true),
            (hex::decode("50").unwrap(), true),
            (hex::decode("31").unwrap(), true),
            (hex::decode("20756c74726963").unwrap(), true),
            (hex::decode("5a3c4ca9b780").unwrap(), false),
            (hex::decode("0706528180").unwrap(), false),
            (hex::decode("6369").unwrap(), true),
            (hex::decode("6e63").unwrap(), true),
            (hex::decode("7365").unwrap(), true),
            (hex::decode("f1e104dedaf8").unwrap(), false),
            (hex::decode("4d").unwrap(), true),
            (hex::decode("2f").unwrap(), true),
            (hex::decode("7469").unwrap(), true),
            (hex::decode("2e204675736365").unwrap(), true),
            (hex::decode("56").unwrap(), true),
            (hex::decode("79").unwrap(), true),
            (hex::decode("30").unwrap(), true),
            (hex::decode("6375").unwrap(), true),
            (hex::decode("20706f72746120").unwrap(), true),
            (hex::decode("2070756c76696e6172").unwrap(), true),
            (hex::decode("39ac809d78").unwrap(), false),
            (hex::decode("53").unwrap(), true),
            (hex::decode("3c2f666f6e743e3c2f").unwrap(), true),
            (hex::decode("2074656d70757320").unwrap(), true),
            (hex::decode("41").unwrap(), true),
            (hex::decode("44").unwrap(), true),
            (hex::decode("34").unwrap(), true),
            (hex::decode("2e2051756973717565").unwrap(), true),
            (hex::decode("2e20496e7465676572").unwrap(), true),
            (hex::decode("2e204d617572697320").unwrap(), true),
            (hex::decode("a26507108780").unwrap(), false),
            (hex::decode("f79420dd5390").unwrap(), false),
            (hex::decode("2067726176696461").unwrap(), true),
            (hex::decode("20626962656e64756d").unwrap(), true),
            (hex::decode("2ddaeaef877380").unwrap(), false),
            (hex::decode("8a44d88f0280").unwrap(), false),
            (hex::decode("2073617069656e20").unwrap(), true),
            (hex::decode("20696163756c6973").unwrap(), true),
            (hex::decode("7ddbbd90b8cc").unwrap(), false),
            (hex::decode("6a").unwrap(), true),
            (hex::decode("55").unwrap(), true),
            (hex::decode("36").unwrap(), true),
            (hex::decode("717565").unwrap(), true),
            (hex::decode("3e203c").unwrap(), true),
            (hex::decode("2e20446f6e656320").unwrap(), true),
            (hex::decode("c096ed2d51c0").unwrap(), false),
            (hex::decode("20766f6c7574706174").unwrap(), true),
            (hex::decode("f767bb3186ff").unwrap(), false),
            (hex::decode("3f7b5734bb5e").unwrap(), false),
            (hex::decode("3f6f054bfc").unwrap(), false),
            (hex::decode("39ac8076d198").unwrap(), false),
            (hex::decode("645e6242e5aaeb").unwrap(), false),
            (hex::decode("49").unwrap(), true),
            (hex::decode("c06799542b8480").unwrap(), false),
            (hex::decode("c060629d8780").unwrap(), false),
            (hex::decode("7320").unwrap(), true),
            (hex::decode("207665686963756c61").unwrap(), true),
            (hex::decode("70a142fbc0").unwrap(), false),
            (hex::decode("206661756369627573").unwrap(), true),
            (hex::decode("2066696e6962757320").unwrap(), true),
            (hex::decode("f7754ea38d80").unwrap(), false),
            (hex::decode("39ad5cea38d8").unwrap(), false),
            (hex::decode("20656c656966656e64").unwrap(), true),
            (hex::decode("6b").unwrap(), true),
            (hex::decode("33").unwrap(), true),
            (hex::decode("45").unwrap(), true),
            (hex::decode("23").unwrap(), true),
            (hex::decode("2e").unwrap(), true),
            (hex::decode("c07da2e508c0").unwrap(), false),
            (hex::decode("70aa544bfbe0").unwrap(), false),
            (hex::decode("3fc783ce62").unwrap(), false),
            (hex::decode("20706f737565726520").unwrap(), true),
            (hex::decode("207072657469756d20").unwrap(), true),
            (hex::decode("63a996dba41d78").unwrap(), false),
            (hex::decode("206c6163696e696120").unwrap(), true),
            (hex::decode("f54de783196a").unwrap(), false),
            (hex::decode("6505155e1ef0").unwrap(), false),
            (hex::decode("fe1ac23c0f00").unwrap(), false),
            (hex::decode("32").unwrap(), true),
            (hex::decode("0178f67fdcb0b62d2fc4d9d05b4ab96f2156b9f31e60").unwrap(), false),
            (hex::decode("3a").unwrap(), true),
            (hex::decode("2b").unwrap(), true),
            (hex::decode("7a").unwrap(), true),
            (hex::decode("0eca7864ed22043b037a660712187bb3de99900d5d981ccd89983a2e170b8c0f00").unwrap(), false),
            (hex::decode("c0b19e933966").unwrap(), false),
            (hex::decode("39ad66a6310a00").unwrap(), false),
            (hex::decode("fd95d4ada780").unwrap(), false),
            (hex::decode("7e6ecea38dbc").unwrap(), false),
            (hex::decode("47").unwrap(), true),
            (hex::decode("5f").unwrap(), true),
            (hex::decode("77").unwrap(), true),
            (hex::decode("4c").unwrap(), true),
            (hex::decode("2d").unwrap(), true),
            (hex::decode("3c3276910202fc4cc1d24078600f0bc008c0f0").unwrap(), false),
            (hex::decode("3cff329d34a07ca53d4d7774100f4f0c9e24a4c5756666ab9ee181e193b48810ec0de9981c0f987bb3de999a57080a201abb30399b1330743e1f0f981e1600").unwrap(), false),
            (hex::decode("1968f064").unwrap(), false),
            (hex::decode("303c327691021d81bd3303890c3dd9ef4ccd2b8405100d5d981ccd89983a2e170b8c0f0cb5feef737652a0001c0e070381c0").unwrap(), false),
            (hex::decode("00e621d3171a9ade9980").unwrap(), false),
            (hex::decode("c02bafde3df7b499ca3201fb738d16ecedb43c").unwrap(), false),
            (hex::decode("f01258f6a538bbc0").unwrap(), false),
            (hex::decode("59").unwrap(), true),
            (hex::decode("3b").unwrap(), true),
            (hex::decode("39").unwrap(), true),
            (hex::decode("54").unwrap(), true),
            (hex::decode("42").unwrap(), true),
            (hex::decode("46").unwrap(), true),
            (hex::decode("3d").unwrap(), true),
      ];

        let huffman_def = HuffmanTable {
            dictionary: dictionary
                .iter()
                .map(|(x, flag)| Some((x.clone(), *flag)))
                .collect(),
            min_codes: [
                0, 0, 0, 0, 4026531840, 3355443200, 2013265920, 1107296256, 570425344, 192937984,
                29360128, 8388608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            max_codes: [
                4294967295, 2147483647, 1073741823, 536870911, 4294967295, 4160749567, 3758096383,
                2885681151, 1996488703, 1283457023, 738197503, 383778815, 196083711, 524287,
                262143, 131071, 65535, 32767, 16383, 8191, 4095, 2047, 1023, 511, 255, 127, 63, 31,
                15, 7, 3, 1, 0,
            ],
            code_dict: [
                (11, false, 2097151),
                (10, false, 4194303),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (10, true, 738197503),
                (9, false, 8388607),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (9, true, 1283457023),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (8, true, 1996488703),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (7, true, 2885681151),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (6, true, 3758096383),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (5, true, 4160749567),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
                (4, true, 4294967295),
            ],
        };

        let mut decoder = HuffmanDecoder { table: huffman_def };

        let data = hex::decode("327b93cc7864f774827864746fd3c321e9d1bf4f0c87af77482786498a07278647cb3df96d4c07f0732f4cf9240601d78929315d5999aae7b8607864ed27196c0de9981c0f987bb3de999a57080a86adbb30399b1330743e1f0f981e001e934ddf91df9624e28e423278903a5fe1d3171a9ade99920061c0e03e482e0072400a088061c0b81f0041c0038220141f00c4848380441ff0125924a39355cf7003988747a5e1d1e9f2c010f6a5e55e483278900c0407d8a6df9bc45dccfab3b6ce84accfa9b26b4ef3121aa41611ed6c77ee32e8efd8a1d1b79c15e3ead81f955b57701f629b634890d527bbb9aaedccdcfc541307110da7699c01e1e3008be0670e7e872351460955371a7b5e5482da2ec8c948fc2a3f938a0782c05e0670e6243549676dfc4faeefd4bfd59db67edc0b2e4646b58edc51f22b9062eee6abb7337c12aa6e09477ee3be6b0dc13bafc8a29c096e91ac7d0f7f289b2e844fdbeacedb0a5d02ab288fa9fb812dd235a6d3aefdd7e6c9f44db9f34c1f51505ae8249d7f2f62d321c45eacedb7ab6180e888e0af1fc485e6e46b623d6bbd06c4f64fb27edb7ce5011595153f5ee71ffc8ae4b0dd974aeafa97c77cc4f5dfbc12da4102a7d9b38a1c84ce088f69e31d33bc862fb6f70454398d27f31e6315a779235b82554dfdd9eee6bf614dcf2483362d405e04100aeab398f31fbafc82ead859d5cd022fdcb0b6f7745efae68cb52ed4609554cc3e873eccf9d02ba62d3315a779b1668d7246b622ee6700aeb3f157d4bcd98949f7bbb9aaedccdec6b15a77988bb99a6081b1a5782554dc7e26087017db782b75f90cb7833e70578d8d032b50313ebba2d71432ece7359d76c59a35cfd455ed5d0d0fd9e704aa9bb228d4cb7cdcb6ddfa208f01d1142a7c439e48c9489256f054bce6b3aee09587b5139fb3c90c100aeb1fd9d2664a44ad79520b6b256792db7611be4d65693d59db67c9bd3b07ed8ed208547f27140f301828eeab23dad399e7289053803c3c6009322973cbff944d90a6a5e1a06c9f975e3c0f46a9be033524eb0bd14abdf26b1a0a7f72c2db04aaa66013b8a1c93acebdb1bb763a26dcf92192dda5aa1011477ecfc55640ce1cc6a9bd303a06a6ec83b23dadb27293ef819c397044b36b2abb5654bfea2a63a26dcf97c9ac37097eebf34da75df6acb52ed40b7f98f31a6d3aeba1ad307bba2fa78e84e1631466980203352c74ceffdcb0b67416065650e6348e8627b03634acfd9e439d5cd031b9616d8255533006a7d4cb6dd20ebf72c2d9929625f35f82554dc6d4398cb9f5cb788c683ba4fd9e5c6d655769d11ad1235b1668d731764c12b0a503944d97147f31e62351ab2526cf930b5c114ade0a97b146696221b5455e99bd682e5fb6f9ca304aaa6604f5aa6ec94c7d4fd0844b0a98d2447b9c01e1e3006a72f9af1242a7f13ebba434aaa661f5157ab60e852e288d0bded5d02da314ed18a334c12b0f4cdeb4172fae5bc46c519a3a36117d4be2b4ef2e28c02badfb62ec94f2342f3eb96f1184ce288cbbea2ac12aa6e27122ca1c50e447635f77db576d7fdd7e5c2de2154d82a0bb7c7ad3ea2ab1a028fc28189f5dd7ed0d812de0cb8a027a6081b1a561c777355db99bfc8ae471439c0b2e46315a80d623dc843640ce1c9f8aa2c7b384f80838c5a67c").unwrap();

        let decoded = decoder.unpack(&data);

        let expected = hex::decode("3c68746d6c3e3c686561643e3c67756964653e3c2f67756964653e3c2f686561643e3c626f64793e3c6e61762020657075623a747970653d22746f63223e203c683120616c69676e3d227269676874223e3c666f6e742073697a653d222b332220666163653d2273616e732d73657269662220636f6c6f723d2223363636223e5461626c65206f6620436f6e74656e74733c2f666f6e743e3c2f68313e203c6f6c203e203c6c692076616c75653d2231223e3c666f6e742073697a653d222b312220666163653d2273616e732d73657269662220636f6c6f723d2223343434223e3c61202066696c65706f733d30303030303030363134203e3c666f6e7420636f6c6f723d2223314541454442223e4368617074657220313c2f666f6e743e3c2f613e3c2f666f6e743e3c2f6c693e203c6c692076616c75653d2232223e3c666f6e742073697a653d222b312220666163653d2273616e732d73657269662220636f6c6f723d2223343434223e3c61202066696c65706f733d30303030303034333930203e3c666f6e7420636f6c6f723d2223314541454442223e4368617074657220323c2f666f6e743e3c2f613e3c2f666f6e743e3c2f6c693e203c6c692076616c75653d2233223e3c666f6e742073697a653d222b312220666163653d2273616e732d73657269662220636f6c6f723d2223343434223e3c61202066696c65706f733d30303030303131303233203e3c666f6e7420636f6c6f723d2223314541454442223e436f707972696768743c2f666f6e743e3c2f613e3c2f666f6e743e3c2f6c693e203c2f6f6c3e203c2f6e61763e20203c6d62703a70616765627265616b2f3e3c683120616c69676e3d227269676874223e3c666f6e742073697a653d222b332220666163653d2273616e732d73657269662220636f6c6f723d2223363636223e4368617074657220313c2f666f6e743e3c2f68313e203c703e3c666f6e742073697a653d222b312220666163653d2273657269662220636f6c6f723d2223343434223e4c6f72656d20697073756d20646f6c6f722073697420616d65742c20636f6e73656374657475722061646970697363696e6720656c69742e204d6f726269206964206c65637475732064696374756d2c206c6f626f727469732075726e6120612c206c7563747573206c696265726f2e20496e746567657220756c74726963696573206e697369206e6563206e69736920677261766964612c2073697420616d65742074656d706f72206469616d20706f73756572652e204d6f72626920636f6e736571756174206c696265726f206672696e67696c6c612070656c6c656e7465737175652076656e656e617469732e20446f6e656320757420706f727461206d657475732e20457469616d20636f6e64696d656e74756d2063757273757320656c69742070756c76696e617220677261766964612e204e756c6c6120636f6e73657175617420696e74657264756d206c656f207365642074696e636964756e742e2053757370656e646973736520706f74656e74692e20496e7465676572207072657469756d20637572737573206c696265726f2c206575206f726e61726520657261742064696374756d2065752e204372617320756c747269636573206d692076656c206f64696f2074656d707573206672696e67696c6c612e204e756c6c6120747269737469717565206e697369206e69736c2c206964207363656c65726973717565207269737573206772617669646120612e2050726f696e206e657175652074656c6c75732c206566666963697475722065752064696374756d2068656e6472657269742c206c6163696e69612076656c206e657175652e2050726f696e2061756775652072697375732c206d6178696d75732061632065726f7320636f6e73656374657475722c2066696e69627573206d6174746973206d657475732e20446f6e65632075742064696374756d206469616d2e20566573746962756c756d2074656d706f72206f72636920696420756c747269636573206c6163696e69612e205072616573656e7420626962656e64756d2c206d6920766974616520657569736d6f6420636f6e64696d656e74756d2c2065726f7320656e696d2074696e636964756e74206f64696f2c2069642073656d7065722065726174206e69736c207669746165206e6973692e204e756e6320636f6e76616c6c6973206d692073697420616d65742076656c69742076656e656e6174697320766573746962756c756d206163206174206d61757269732e3c2f666f6e743e3c2f703e203c703e3c666f6e742073697a653d222b312220666163653d2273657269662220636f6c6f723d2223343434223e4475697320766573746962756c756d20656c6974206e6f6e20746f72746f72206c756374757320756c7472696365732e204e756c6c6120666163696c6973692e20416c697175616d2072686f6e6375732c2065737420617420656c656d656e74756d2073656d7065722c20746f72746f72207269737573207072657469756d206469616d2c206567657420636f6e64696d656e74756d20697073756d2066656c697320656765742073617069656e2e204e756c6c616d20656c69742075726e612c206c6163696e69612061206d65747573206c75637475732c2076617269757320756c747269636573206c65637475732e20496e2070656c6c656e74657371756520657261742073697420616d6574206d61676e61206672696e67696c6c612c20696e206c7563747573206c656374757320616363756d73616e2e204e756c6c612074696e636964756e74206469676e697373696d20656c656966656e642e2044756973206c6163696e696120726973757320666163696c69736973206c616375732074656d706f722c20657420666163696c6973697320657374206f726e6172652e20517569737175652076656c2073656d20636f6e6775652c207661726975732065726f7320696e2c20636f6e7365717561742070757275732e204e756c6c6120736f64616c657320696420647569207574206665726d656e74756d2e20566573746962756c756d20736564206a7573746f207365642073617069656e20706c61636572617420636f6e6775652e204e616d20696e20626962656e64756d2070757275732e3c2f666f6e743e3c2f703e203c703e3c666f6e742073697a653d222b312220666163653d2273657269662220636f6c6f723d2223343434223e4c6f72656d20697073756d20646f6c6f722073697420616d65742c20636f6e73656374657475722061646970697363696e6720656c69742e20496e2074696e636964756e742c20657820736564206c6f626f7274697320696e74657264756d2c20707572757320647569206665726d656e74756d2065782c206e6f6e2064696374756d206578207175616d206575206e756e632e2053757370656e646973736520706f74656e74692e2041656e65616e20756c7472696369657320636f6d6d6f646f2068656e6472657269742e20457469616d207472697374697175652073616769747469732074656c6c75732e20416c697175616d206572617420766f6c75747061742e204d616563656e617320766974616520747269737469717565206e756c6c612e2053656420766974616520766f6c75747061742065782c2065742068656e64726572697420697073756d2e204e756c6c616d207665686963756c612076656c6974207669746165207665686963756c61206d6178696d75732e3c2f666f6e743e3c2f703e203c703e3c666f6e742073697a653d222b312220666163653d2273657269662220636f6c6f723d2223343434223e5072616573656e74207574206e6973692070656c6c656e7465737175652c20637572737573206e756c6c612061632c2076756c707574617465206c656f2e20467573636520756c747269636965732c206d61676e612063757273757320766573746962756c756d20766976657272612c20657261742065726f73206d6178696d75732065782c20696420666175636962757320726973757320617567756520696e2073617069656e2e205175697371756520746f72746f722061756775652c2070756c76696e61722061632066656c69732061742c20626962656e64756d20736f6c6c696369747564696e206d617373612e204c6f72656d20697073756d20646f6c6f722073697420616d65742c20636f6e73656374657475722061646970697363696e6720656c69742e20536564206d617572697320697073756d2c20736f6c6c696369747564696e20666163696c697369732076656c6974206e6f6e2c2074656d70757320616363756d73616e2070757275732e20446f6e6563206567657420706861726574726120697073756d2e204e756c6c616d2076756c707574617465206d616c65737561646120697073756d20736564206d6f6c6c69732e204e756c6c6120706f72746120656c6974206e6563206567657374617320636f6e7365717561742e204675736365206120707572757320766976657272612c20756c7472696365732073617069656e20696e2c20766172697573206c656f2e204e616d2061742074656c6c75732076656c20746f72746f722070756c76696e61722070756c76696e6172207365642073697420616d65742065782e205175697371756520766573746962756c756d20706c616365726174206d617373612c20757420706f7274612065726f7320626c616e64697420657569736d6f642e204e756c6c616d207669746165206e756c6c6120736564206e6571756520656c656966656e64206c6f626f7274697320736564206e6f6e206c616375732e20457469616d2076756c707574617465206d6f6c6c69732073617069656e20736564206f726e6172652e3c2f666f6e743e3c2f703e203c703e3c666f6e742073697a653d222b312220666163653d2273657269662220636f6c6f723d2223343434223e4e756c6c616d2065726f73206469616d2c2068656e6472657269742076656c206e69626820696e2c206d616c65737561646120616c6971756574206d617373612e204e616d20626c616e6469742065676573746173206d617373612c2073697420616d6574206566666963697475722065726174206c75637475732076656c2e20496e206575206c656f206174206e69626820656765737461732076656e656e617469732076656c206e65632065726f732e204e756c6c6120626962656e64756d2073617069656e2076656c2076656c697420696163756c69732c20696e206665756769617420726973757320766573746962756c756d2e2053757370656e646973736520706c616365726174206c616f726565742065726f732c20657420756c6c616d636f7270657220657374206f726e6172652065752e205072616573656e7420696d70657264696574206c616375732076656c207665686963756c6120616363756d73616e2e20446f6e6563206672696e67696c6c61206f64696f2076656c69742e20437261732074656d7075732065737420696e206c6163757320656c656966656e642c206375727375732070656c6c656e746573717565206c6f72656d207665686963756c612e205574206d6574757320").unwrap();

        assert_eq!(decoded, expected);

        println!(
            "compression ratio: {:?}",
            data.len() as f32 / expected.len() as f32
        );
    }
}
