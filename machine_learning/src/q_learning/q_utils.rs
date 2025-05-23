use env::Env;
use env::rand;
use std::collections::HashMap;

pub fn estimate_max_q_value<E: Env>(
    env: &E,
    q_table: &HashMap<(E::State, E::Action), f32>,
    next_state: E::State,
    prev_action: E::Action,
) -> f32
where
    E::State: Copy + Eq + std::hash::Hash,
    E::Action: Copy + Eq + std::hash::Hash,
{
    if env.is_terminal() {
        return 0.0;
    }

    let actions = env.legal_actions(Some((next_state, prev_action)));
    if actions.is_empty() {
        return 0.0;
    }

    actions
        .iter()
        .map(|&a| *q_table.get(&(next_state, a)).unwrap_or(&0.0))
        .filter(|q| q.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

pub fn random_action<E: Env>(
    actions: &[E::Action],
    state: E::State,
    rng: &mut impl rand::Rng,
    exported_table: &HashMap<(E::State, E::Action), usize>,
) -> E::Action
where
    E::State: Copy + Eq + std::hash::Hash,
    E::Action: Copy + Eq + std::hash::Hash,
{
    if actions.is_empty() {
        panic!("No legal actions available");
    } else if actions.len() == 1 {
        return actions[0];
    }

    let result: Vec<(<E as Env>::Action, usize)> = actions.iter().map(|&a| {
        let q = *exported_table.get(&(state, a)).unwrap_or(&0);
        (a, q)
    }).collect::<Vec<_>>();
    let min_q = result.iter().map(|&(_, q)| q).min().unwrap_or(0);
    let min_actions: Vec<_> = result
        .iter()
        .filter(|&&(_, q)| q == min_q)
        .map(|&(a, _)| a)
        .collect();
    if min_actions.is_empty() {
        return actions[rng.random_range(0..actions.len())];
    } else if min_actions.len() == 1 {
        return min_actions[0];
    } else {
        return min_actions[rng.random_range(0..min_actions.len())];
    }
}

pub fn estimate_double_q_target<E: Env>(
    env: &E,
    q_select: &HashMap<(E::State, E::Action), f32>,
    q_eval: &HashMap<(E::State, E::Action), f32>,
    next_state: E::State,
    prev_action: E::Action,
) -> f32
where
    E::State: Copy + Eq + std::hash::Hash,
    E::Action: Copy + Eq + std::hash::Hash,
{
    if env.is_terminal() {
        return 0.0;
    }

    let actions = env.legal_actions(Some((next_state, prev_action)));
    if actions.is_empty() {
        return 0.0;
    }

    let a_prime = actions.iter().copied().max_by(|&a1, &a2| {
        let q1 = *q_select.get(&(next_state, a1)).unwrap_or(&0.0);
        let q2 = *q_select.get(&(next_state, a2)).unwrap_or(&0.0);
        q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
    });

    a_prime
        .map(|a| *q_eval.get(&(next_state, a)).unwrap_or(&0.0))
        .unwrap_or(0.0)
}
