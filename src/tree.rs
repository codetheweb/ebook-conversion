use std::{
    collections::{BinaryHeap, HashMap},
    fmt::Debug,
    hash::Hash,
};

use bitvec::vec::BitVec;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct HuffmanNode<Symbol> {
    symbol: Option<Symbol>,
    frequency: usize,
    left: Option<Box<HuffmanNode<Symbol>>>,
    right: Option<Box<HuffmanNode<Symbol>>>,
}

impl<Symbol> PartialOrd for HuffmanNode<Symbol>
where
    Symbol: Eq + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Symbol> Ord for HuffmanNode<Symbol>
where
    Symbol: Eq + Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .frequency
            .cmp(&self.frequency)
            .then(self.symbol.cmp(&other.symbol))
    }
}

impl<Symbol> HuffmanNode<Symbol> {
    fn leaf(symbol: Symbol, frequency: usize) -> Self {
        Self {
            symbol: Some(symbol),
            frequency: frequency,
            left: None,
            right: None,
        }
    }

    fn node(frequency: usize, left: HuffmanNode<Symbol>, right: HuffmanNode<Symbol>) -> Self {
        Self {
            symbol: None,
            frequency: frequency,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
}

impl<Symbol> FromIterator<(Symbol, usize)> for HuffmanNode<Symbol>
where
    Symbol: Eq + Ord + Clone,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Symbol, usize)>,
    {
        let mut queue = BinaryHeap::new();

        for (symbol, frequency) in iter {
            queue.push(Self::leaf(symbol, frequency));
        }

        while queue.len() > 1 {
            if let (Some(left), Some(right)) = (queue.pop(), queue.pop()) {
                queue.push(Self::node(left.frequency + right.frequency, left, right))
            }
        }

        queue.pop().unwrap()
    }
}

impl<Symbol> HuffmanNode<Symbol>
where
    Symbol: Eq + Ord + Clone,
{
    pub fn push_deeper(&mut self, levels: usize) {
        if self.symbol.is_none() {
            // todo: error or panic?
            panic!("Can't push deeper on a non-leaf node");
        }

        let original_node = self.clone();
        let frequency = self.frequency;

        let mut current_node = self;
        for i in 0..levels {
            let new_node = Self {
                symbol: None,
                frequency,
                left: None,
                right: None,
            };

            if i == levels - 1 {
                current_node.left = Some(Box::new(original_node.clone()));
            } else {
                current_node.left = Some(Box::new(new_node));
            }
            current_node.right = None;
            current_node.symbol = None;
            current_node = current_node.left.as_mut().unwrap();
        }
    }
}

#[derive(Debug)]
pub struct HuffmanTree<Symbol> {
    root: Box<HuffmanNode<Symbol>>,
}

impl<Symbol> FromIterator<(Symbol, usize)> for HuffmanTree<Symbol>
where
    Symbol: Eq + Ord + Hash + Clone,
{
    fn from_iter<I>(frequencies: I) -> Self
    where
        I: IntoIterator<Item = (Symbol, usize)>,
    {
        Self {
            root: Box::new(HuffmanNode::from_iter(frequencies)),
        }
    }
}

impl<Symbol> HuffmanTree<Symbol>
where
    Symbol: Eq + Ord + Hash + Clone,
{
    // todo: balance more intelligently
    pub fn enforce_minimum_code_length(&mut self, min_code_length: usize) {
        let codebook = self.get_codebook();

        for (_, bits) in codebook.iter() {
            if bits.len() < min_code_length {
                let node = self.code_to_node_mut(bits.clone()).unwrap();
                node.push_deeper(min_code_length - bits.len());
            }
        }
    }

    pub fn get_codebook(&self) -> HashMap<Symbol, BitVec> {
        fn collect<Symbol>(
            codebook: &mut HashMap<Symbol, BitVec>,
            node: &HuffmanNode<Symbol>,
            current_bits: BitVec,
        ) where
            Symbol: Eq + Hash + Clone,
        {
            if let Some(symbol) = &node.symbol {
                codebook.insert(symbol.clone(), current_bits);
            } else {
                if let Some(left) = node.left.as_ref() {
                    let mut left_bits = current_bits.clone();
                    left_bits.push(false);
                    collect(codebook, left, left_bits);
                }

                if let Some(right) = node.right.as_ref() {
                    let mut right_bits = current_bits;
                    right_bits.push(true);
                    collect(codebook, right, right_bits);
                }
            }
        }

        let mut codebook = HashMap::new();
        collect(&mut codebook, &self.root, BitVec::new());

        codebook
    }

    fn code_to_node_mut(&mut self, code: BitVec) -> Option<&mut HuffmanNode<Symbol>> {
        let mut current_node = &mut *self.root;

        for bit in code {
            current_node = if bit {
                current_node.right.as_mut()?
            } else {
                current_node.left.as_mut()?
            };
        }

        Some(current_node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_node_from_iter() {
        let frequencies = vec![('a', 4), ('b', 2), ('c', 1), ('d', 1)];
        let tree = HuffmanTree::from_iter(frequencies.into_iter());

        let codebook = tree.get_codebook();
        // 'a' is the most frequent, so it should be encoded with a single bit
        assert_eq!(codebook.get(&'a').unwrap().len(), 1);
    }

    #[test]
    fn test_huffman_tree_min_code_length() {
        let frequencies = vec![('a', 4), ('b', 2), ('c', 1), ('d', 1)];
        let mut tree = HuffmanTree::from_iter(frequencies.into_iter());
        tree.enforce_minimum_code_length(4);

        let codebook = tree.get_codebook();

        for (_, bits) in codebook.iter() {
            assert!(bits.len() >= 4);
        }
    }
}
