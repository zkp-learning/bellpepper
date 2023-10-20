use std::collections::BTreeMap;
use std::ops::{Add, Sub};

use ff::PrimeField;
use serde::{Deserialize, Serialize};

/// Represents a variable in our constraint system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable(pub Index);

impl Variable {
    /// This constructs a variable with an arbitrary index.
    /// Circuit implementations are not recommended to use this.
    pub fn new_unchecked(idx: Index) -> Variable {
        Variable(idx)
    }

    /// This returns the index underlying the variable.
    /// Circuit implementations are not recommended to use this.
    pub fn get_unchecked(&self) -> Index {
        self.0
    }
}

/// Represents the index of either an input variable or
/// auxiliary variable.
#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash, Serialize, Deserialize)]
pub enum Index {
    Input(usize),
    Aux(usize),
}

/// This represents a linear combination of some variables, with coefficients
/// in the scalar field of a pairing-friendly elliptic curve group.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCombination<Scalar: PrimeField> {
    inputs: Indexer<Scalar>,
    aux: Indexer<Scalar>,
}

#[derive(Clone, Debug, PartialEq)]
struct Indexer<T> {
    /// Stores a list of `T` indexed by the number in the first slot of the tuple.
    values: BTreeMap<usize, T>,
}

impl<T> Default for Indexer<T> {
    fn default() -> Self {
        Indexer {
            values: BTreeMap::new(),
        }
    }
}

impl<T> Indexer<T> {
    pub fn from_value(index: usize, value: T) -> Self {
        let mut temp = BTreeMap::new();
        temp.insert(index, value);
        Indexer { values: temp }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &T)> + '_ {
        self.values.iter().map(|(key, value)| (key, value))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut T)> + '_ {
        self.values.iter_mut()
    }

    pub fn insert_or_update<F, G>(&mut self, key: usize, insert: F, update: G)
    where
        F: FnOnce() -> T,
        G: FnOnce(&mut T),
    {
        self.values
            .entry(key)
            .and_modify(|v| update(v))
            .or_insert(insert());
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<Scalar: PrimeField> Default for LinearCombination<Scalar> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<Scalar: PrimeField> LinearCombination<Scalar> {
    pub fn zero() -> LinearCombination<Scalar> {
        LinearCombination {
            inputs: Default::default(),
            aux: Default::default(),
        }
    }

    pub fn from_coeff(var: Variable, coeff: Scalar) -> Self {
        match var {
            Variable(Index::Input(i)) => Self {
                inputs: Indexer::from_value(i, coeff),
                aux: Default::default(),
            },
            Variable(Index::Aux(i)) => Self {
                inputs: Default::default(),
                aux: Indexer::from_value(i, coeff),
            },
        }
    }

    pub fn from_variable(var: Variable) -> Self {
        Self::from_coeff(var, Scalar::ONE)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Variable, &Scalar)> + '_ {
        self.inputs
            .iter()
            .map(|(k, v)| (Variable(Index::Input(*k)), v))
            .chain(self.aux.iter().map(|(k, v)| (Variable(Index::Aux(*k)), v)))
    }

    #[inline]
    pub fn iter_inputs(&self) -> impl Iterator<Item = (&usize, &Scalar)> + '_ {
        self.inputs.iter()
    }

    #[inline]
    pub fn iter_aux(&self) -> impl Iterator<Item = (&usize, &Scalar)> + '_ {
        self.aux.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Variable, &mut Scalar)> + '_ {
        self.inputs
            .iter_mut()
            .map(|(k, v)| (Variable(Index::Input(*k)), v))
            .chain(
                self.aux
                    .iter_mut()
                    .map(|(k, v)| (Variable(Index::Aux(*k)), v)),
            )
    }

    #[inline]
    fn add_assign_unsimplified_input(&mut self, new_var: usize, coeff: Scalar) {
        self.inputs
            .insert_or_update(new_var, || coeff, |val| *val += coeff);
    }

    #[inline]
    fn add_assign_unsimplified_aux(&mut self, new_var: usize, coeff: Scalar) {
        self.aux
            .insert_or_update(new_var, || coeff, |val| *val += coeff);
    }

    pub fn add_unsimplified(
        mut self,
        (coeff, var): (Scalar, Variable),
    ) -> LinearCombination<Scalar> {
        match var.0 {
            Index::Input(new_var) => {
                self.add_assign_unsimplified_input(new_var, coeff);
            }
            Index::Aux(new_var) => {
                self.add_assign_unsimplified_aux(new_var, coeff);
            }
        }

        self
    }

    #[inline]
    fn sub_assign_unsimplified_input(&mut self, new_var: usize, coeff: Scalar) {
        self.add_assign_unsimplified_input(new_var, -coeff);
    }

    #[inline]
    fn sub_assign_unsimplified_aux(&mut self, new_var: usize, coeff: Scalar) {
        self.add_assign_unsimplified_aux(new_var, -coeff);
    }

    pub fn sub_unsimplified(
        mut self,
        (coeff, var): (Scalar, Variable),
    ) -> LinearCombination<Scalar> {
        match var.0 {
            Index::Input(new_var) => {
                self.sub_assign_unsimplified_input(new_var, coeff);
            }
            Index::Aux(new_var) => {
                self.sub_assign_unsimplified_aux(new_var, coeff);
            }
        }

        self
    }

    pub fn len(&self) -> usize {
        self.inputs.len() + self.aux.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.aux.is_empty()
    }

    pub fn eval(&self, input_assignment: &[Scalar], aux_assignment: &[Scalar]) -> Scalar {
        let mut acc = Scalar::ZERO;

        let one = Scalar::ONE;

        for (index, coeff) in self.iter_inputs() {
            let mut tmp = input_assignment[*index];
            if coeff != &one {
                tmp *= coeff;
            }
            acc += tmp;
        }

        for (index, coeff) in self.iter_aux() {
            let mut tmp = aux_assignment[*index];
            if coeff != &one {
                tmp *= coeff;
            }
            acc += tmp;
        }

        acc
    }
}

impl<Scalar: PrimeField> Add<(Scalar, Variable)> for LinearCombination<Scalar> {
    type Output = LinearCombination<Scalar>;

    fn add(self, (coeff, var): (Scalar, Variable)) -> LinearCombination<Scalar> {
        self.add_unsimplified((coeff, var))
    }
}

impl<Scalar: PrimeField> Sub<(Scalar, Variable)> for LinearCombination<Scalar> {
    type Output = LinearCombination<Scalar>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, (coeff, var): (Scalar, Variable)) -> LinearCombination<Scalar> {
        self.sub_unsimplified((coeff, var))
    }
}

impl<Scalar: PrimeField> Add<Variable> for LinearCombination<Scalar> {
    type Output = LinearCombination<Scalar>;

    fn add(self, other: Variable) -> LinearCombination<Scalar> {
        self + (Scalar::ONE, other)
    }
}

impl<Scalar: PrimeField> Sub<Variable> for LinearCombination<Scalar> {
    type Output = LinearCombination<Scalar>;

    fn sub(self, other: Variable) -> LinearCombination<Scalar> {
        self - (Scalar::ONE, other)
    }
}

impl<'a, Scalar: PrimeField> Add<&'a LinearCombination<Scalar>> for LinearCombination<Scalar> {
    type Output = LinearCombination<Scalar>;

    fn add(mut self, other: &'a LinearCombination<Scalar>) -> LinearCombination<Scalar> {
        for (var, val) in other.inputs.iter() {
            self.add_assign_unsimplified_input(*var, *val);
        }

        for (var, val) in other.aux.iter() {
            self.add_assign_unsimplified_aux(*var, *val);
        }

        self
    }
}

impl<'a, Scalar: PrimeField> Sub<&'a LinearCombination<Scalar>> for LinearCombination<Scalar> {
    type Output = LinearCombination<Scalar>;

    fn sub(mut self, other: &'a LinearCombination<Scalar>) -> LinearCombination<Scalar> {
        for (var, val) in other.inputs.iter() {
            self.sub_assign_unsimplified_input(*var, *val);
        }

        for (var, val) in other.aux.iter() {
            self.sub_assign_unsimplified_aux(*var, *val);
        }

        self
    }
}

impl<'a, Scalar: PrimeField> Add<(Scalar, &'a LinearCombination<Scalar>)>
    for LinearCombination<Scalar>
{
    type Output = LinearCombination<Scalar>;

    fn add(
        mut self,
        (coeff, other): (Scalar, &'a LinearCombination<Scalar>),
    ) -> LinearCombination<Scalar> {
        for (var, val) in other.inputs.iter() {
            self.add_assign_unsimplified_input(*var, *val * coeff);
        }

        for (var, val) in other.aux.iter() {
            self.add_assign_unsimplified_aux(*var, *val * coeff);
        }

        self
    }
}

impl<'a, Scalar: PrimeField> Sub<(Scalar, &'a LinearCombination<Scalar>)>
    for LinearCombination<Scalar>
{
    type Output = LinearCombination<Scalar>;

    fn sub(
        mut self,
        (coeff, other): (Scalar, &'a LinearCombination<Scalar>),
    ) -> LinearCombination<Scalar> {
        for (var, val) in other.inputs.iter() {
            self.sub_assign_unsimplified_input(*var, *val * coeff);
        }

        for (var, val) in other.aux.iter() {
            self.sub_assign_unsimplified_aux(*var, *val * coeff);
        }

        self
    }
}

#[cfg(all(test))]
mod tests {
    use super::*;
    use blstrs::Scalar;
    use ff::Field;

    #[test]
    fn test_add_simplify() {
        let n = 5;

        let mut lc = LinearCombination::<Scalar>::zero();

        let mut expected_sums = vec![Scalar::ZERO; n];
        let mut total_additions = 0;
        for (i, expected_sum) in expected_sums.iter_mut().enumerate() {
            for _ in 0..i + 1 {
                let coeff = Scalar::ONE;
                lc = lc + (coeff, Variable::new_unchecked(Index::Aux(i)));
                *expected_sum += coeff;
                total_additions += 1;
            }
        }

        // There are only as many terms as distinct variable Indexes — not one per addition operation.
        assert_eq!(n, lc.len());
        assert!(lc.len() != total_additions);

        // Each variable has the expected coefficient, the sume of those added by its Index.
        lc.iter().for_each(|(var, coeff)| match var.0 {
            Index::Aux(i) => assert_eq!(expected_sums[i], *coeff),
            _ => panic!("unexpected variable type"),
        });
    }

    #[test]
    fn test_insert_or_update() {
        let mut indexer = Indexer::default();
        let one = Scalar::ONE;
        let mut two = one;
        two += one;

        indexer.insert_or_update(2, || one, |v| *v += one);
        assert_eq!(
            &indexer.values.clone().into_iter().collect::<Vec<_>>(),
            &[(2, one)]
        );

        indexer.insert_or_update(3, || one, |v| *v += one);
        assert_eq!(
            &indexer.values.clone().into_iter().collect::<Vec<_>>(),
            &[(2, one), (3, one)]
        );

        indexer.insert_or_update(1, || one, |v| *v += one);
        assert_eq!(
            &indexer.values.clone().into_iter().collect::<Vec<_>>(),
            &[(1, one), (2, one), (3, one)]
        );

        indexer.insert_or_update(2, || one, |v| *v += one);
        assert_eq!(
            &indexer.values.into_iter().collect::<Vec<_>>(),
            &[(1, one), (2, two), (3, one)]
        );
    }

    #[test]
    fn test_eval() {
        let mut lc = LinearCombination::<Scalar>::zero();
        for i in 0..10 {
            lc = lc
                + (
                    Scalar::from(i + 1),
                    Variable::new_unchecked(Index::Input(i as usize)),
                );
            lc = lc
                + (
                    Scalar::from(i + 10),
                    Variable::new_unchecked(Index::Aux(i as usize)),
                );
        }
        let mut input_assignment = Vec::new();
        let mut aux_assignment = Vec::new();

        for i in 0..10 {
            input_assignment.push(Scalar::from(i + 1));
            aux_assignment.push(Scalar::from(i + 2));
        }
        let val = lc.eval(&input_assignment, &aux_assignment);
        let mut sum = 0;
        for i in 0..10 {
            sum += (i + 1) * (i + 1);
            sum += (i + 2) * (i + 10);
        }
        assert_eq!(val, Scalar::from(sum));

        println!("{}", val);
        println!("{:?}", sum);
    }
}
