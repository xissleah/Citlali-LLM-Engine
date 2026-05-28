// crates/citlali-runtime/src/sampler.rs
//! 采样策略：greedy, temperature, top-k, top-p, repetition penalty

use rand::Rng;

/// 生成参数配置
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 最大生成 token 数
    pub max_tokens: usize,
    /// 停止 token IDs（遇到即停止生成）
    pub stop_tokens: Vec<u32>,
    /// Temperature: 0.0 = greedy, >0 增加随机性（推荐 0.6~1.0）
    pub temperature: f32,
    /// Top-K: 只从概率最高的 K 个 token 中采样（0 = 不限制）
    pub top_k: usize,
    /// Top-P (Nucleus): 从累积概率达到 P 的最小集合中采样（1.0 = 不限制）
    pub top_p: f32,
    /// Repetition Penalty: >1.0 惩罚重复 token（推荐 1.0~1.3）
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            // 默认空，由调用方从 tokenizer.stop_token_ids 设置
            stop_tokens: Vec::new(),
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

impl SamplerConfig {
    /// Greedy 配置（确定性输出，适合代码生成/测试）
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    /// 核心采样函数：根据配置从 logits 中采样一个 token
    /// `past_tokens`: 已生成的 token 序列，用于 repetition penalty
    pub fn sample(&self, logits: &[f32], past_tokens: &[u32]) -> u32 {
        // temperature=0 等价于 greedy
        if self.temperature == 0.0 {
            return argmax(logits);
        }

        let mut scores: Vec<f32> = logits.to_vec();

        // 1. Repetition Penalty
        if self.repetition_penalty != 1.0 {
            apply_repetition_penalty(&mut scores, past_tokens, self.repetition_penalty);
        }

        // 2. Temperature scaling
        let inv_temp = 1.0 / self.temperature;
        for s in scores.iter_mut() {
            *s *= inv_temp;
        }

        // 3. Top-K filtering
        let candidates = top_k_filter(&scores, self.top_k);

        // 4. Top-P (nucleus) filtering + sampling
        sample_top_p(&candidates, self.top_p)
    }
}

/// Greedy 采样：取 logits 最大值的 index
pub fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as u32
}

/// Repetition Penalty: 对已出现的 token 的 logit 进行惩罚
/// 正 logit 除以 penalty，负 logit 乘以 penalty
fn apply_repetition_penalty(scores: &mut [f32], past_tokens: &[u32], penalty: f32) {
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < scores.len() {
            if scores[idx] > 0.0 {
                scores[idx] /= penalty;
            } else {
                scores[idx] *= penalty;
            }
        }
    }
}

/// Top-K 过滤：返回 (token_id, score) 对，按 score 降序
/// top_k=0 表示不过滤，返回全部
fn top_k_filter(scores: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i as u32, s))
        .collect();

    // 按 score 降序排列
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }
    indexed
}

/// Top-P (Nucleus) 采样：从累积概率达到 p 的最小集合中采样
/// 输入已按 score 降序排列
fn sample_top_p(candidates: &[(u32, f32)], top_p: f32) -> u32 {
    if candidates.is_empty() {
        return 0;
    }

    // Softmax
    let max_score = candidates[0].1;
    let exps: Vec<f32> = candidates.iter().map(|(_, s)| (s - max_score).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // 确定 nucleus 边界
    let mut cumulative = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // 在 cutoff 范围内重新归一化并采样
    let nucleus_sum: f32 = probs[..cutoff].iter().sum();
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * nucleus_sum;

    let mut acc = 0.0f32;
    for i in 0..cutoff {
        acc += probs[i];
        if acc >= r {
            return candidates[i].0;
        }
    }

    // fallback
    candidates[0].0
}
