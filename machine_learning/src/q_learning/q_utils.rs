use env::Env;
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
