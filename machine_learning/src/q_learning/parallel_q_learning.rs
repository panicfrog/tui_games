use super::q_utils::estimate_max_q_value;
use env::rand::Rng;
use env::Env;
use rayon::prelude::*;
use std::collections::HashMap;

/// 多线程 Q-learning，分周期同步
pub fn rayon_parallel_q_learning<E>(
    env: &E,
    episodes: usize,
    max_steps: usize,
    alpha: f32,
    gamma: f32,
    num_workers: usize, // 线程数（如cpu核心数 - 1）
    num_cycles: usize,  // 同步周期数（如5）
) -> HashMap<(E::State, E::Action), f32>
where
    E::State: Copy + Eq + std::hash::Hash + std::fmt::Debug + Send + 'static,
    E::Action: Copy + Eq + std::hash::Hash + std::fmt::Debug + Send + 'static,
    E: Env + Clone + Send + 'static,
{
    let epslion_segment_per_cycle = 1.0 / (num_cycles as f32);
    let episodes_per_cycle = episodes / num_cycles;
    let episodes_per_worker_per_cycle = episodes_per_cycle / num_workers;

    // 初始化全局 Q 表
    let mut global_q_table: HashMap<(E::State, E::Action), (f32, usize)> = HashMap::new();

    for cycle in 0..num_cycles {
        let current_base_epsilon = epslion_segment_per_cycle * (cycle as f32);
        // 各worker并行训练各自Q表
        let envs: Vec<_> = (0..num_workers)
            .map(|_| (env.clone(), global_q_table.clone()))
            .collect();
        let worker_tables: Vec<HashMap<(E::State, E::Action), (f32, usize)>> = envs
            .into_par_iter()
            .map(|(env, q_table)| {
                let mut local_env = env.clone();
                let mut q_table = q_table.clone();

                let mut rng = env::rand::rng();
                for episode in 0..episodes_per_worker_per_cycle {
                    let mut state = local_env.reset();
                    let epsilon = 1.0 - current_base_epsilon - (episode as f32) / (episodes_per_worker_per_cycle as f32) * epslion_segment_per_cycle; 

                    for _ in 0..max_steps {
                        let actions = local_env.legal_actions(None);
                        if actions.is_empty() {
                            break;
                        }

                        let action = if actions.len() == 1 {
                            actions[0]
                        } else if rng.random::<f32>() < epsilon {
                            actions[rng.random_range(0..actions.len())]
                        } else {
                            actions
                                .iter()
                                .copied()
                                .max_by(|&a1, &a2| {
                                    let q1 =
                                        q_table.get(&(state, a1)).map(|&(v, _)| v).unwrap_or(0.0);
                                    let q2 =
                                        q_table.get(&(state, a2)).map(|&(v, _)| v).unwrap_or(0.0);
                                    q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .unwrap_or_else(|| actions[rng.random_range(0..actions.len())])
                        };

                        let (next_state, _) = local_env.step(action);
                        let reward = if local_env.is_win() {
                            10.0
                        } else if !local_env.is_terminal() {
                            -0.01
                        } else {
                            -1.0
                        };

                        // 只用本地表做max
                        let target = estimate_max_q_value(
                            &local_env,
                            &q_table.iter().map(|(&k, &(v, _))| (k, v)).collect(),
                            next_state,
                            action,
                        );

                        let entry = q_table.entry((state, action)).or_insert((0.0, 0));
                        entry.0 += alpha * (reward + gamma * target - entry.0);
                        entry.1 += 1;
                        state = next_state;

                        if local_env.is_win() || local_env.is_terminal() {
                            break;
                        }
                    }
                }

                q_table
            })
            .collect();

        // 全局 Q 表合并采样加权平均
        let mut merged: HashMap<(E::State, E::Action), (f32, usize)> = HashMap::new();
        for q_table in worker_tables {
            for (key, (value, count)) in q_table {
                merged
                    .entry(key)
                    .and_modify(|(sum, total_count)| {
                        if count > 0 {
                            *sum += value as f32;
                            *total_count += 1;
                        }
                    })
                    .or_insert((0.0, 0));
            }
        }
        // 归一化
        for (_key, (sum, total_count)) in merged.iter_mut() {
            if *total_count > 0 {
                *sum /= *total_count as f32;
                *total_count = 0;
            }
        }

        global_q_table = merged;
    }

    // 输出最终Q表（去掉计数，只保留 Q 值）
    global_q_table
        .into_iter()
        .map(|(k, (v, _))| (k, v))
        .collect()
}
