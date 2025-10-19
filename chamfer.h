#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) ||           \
    defined(__SSSE3__) || defined(__SSE3__)
#if !defined(__riscv)
#include <immintrin.h>
#endif
#endif
#endif
#endif

class Chamfer {
private:
  const float *train;
  const uint32_t *train_counts;
  int vec_dim;
  int num_train_points;
  std::vector<uint32_t> train_offsets;

public:
  Chamfer(const float *train, const uint32_t *train_counts, int vec_dim,
          int num_train_points)
      : train(train), train_counts(train_counts), vec_dim(vec_dim),
        num_train_points(num_train_points) {

    train_offsets.resize(num_train_points);
    uint32_t offset = 0;
    for (int i = 0; i < num_train_points; i++) {
      train_offsets[i] = offset;
      offset += train_counts[i];
    }
  }

  float compute_chamfer_similarity(const float *q, uint32_t count,
                                   int train_point_idx) const {
    const uint32_t train_offset = train_offsets[train_point_idx];
    const uint32_t train_count = train_counts[train_point_idx];

    alignas(64) float max_buf_static[64];
    float *max_dot = (count <= 64) ? max_buf_static : new float[count];
    for (uint32_t i = 0; i < count; ++i)
      max_dot[i] = -std::numeric_limits<float>::infinity();

    for (uint32_t j = 0; j < train_count; ++j) {
      const float *tj = train + (train_offset + j) * vec_dim;

#if defined(__AVX512F__)
      uint32_t i = 0;
      for (; i + 7 < count; i += 8) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();

        const float *q0 = q + (i + 0) * vec_dim;
        const float *q1 = q + (i + 1) * vec_dim;
        const float *q2 = q + (i + 2) * vec_dim;
        const float *q3 = q + (i + 3) * vec_dim;
        const float *q4 = q + (i + 4) * vec_dim;
        const float *q5 = q + (i + 5) * vec_dim;
        const float *q6 = q + (i + 6) * vec_dim;
        const float *q7 = q + (i + 7) * vec_dim;

        int k = 0;
        for (; k + 15 < vec_dim; k += 16) {
          __m512 t = _mm512_loadu_ps(tj + k);
          acc0 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q0 + k), acc0);
          acc1 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q1 + k), acc1);
          acc2 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q2 + k), acc2);
          acc3 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q3 + k), acc3);
          acc4 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q4 + k), acc4);
          acc5 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q5 + k), acc5);
          acc6 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q6 + k), acc6);
          acc7 = _mm512_fmadd_ps(t, _mm512_loadu_ps(q7 + k), acc7);

          _mm_prefetch(reinterpret_cast<const char *>(tj + k + 64),
                       _MM_HINT_T0);
        }

        auto hsum512 = [](__m512 x) -> float {
          __m256 lo = _mm512_castps512_ps256(x);
          __m256 hi = _mm512_extractf32x8_ps(x, 1);
          __m256 s = _mm256_add_ps(lo, hi);
          __m128 lo128 = _mm256_castps256_ps128(s);
          __m128 hi128 = _mm256_extractf128_ps(s, 1);
          __m128 s128 = _mm_add_ps(lo128, hi128);
          __m128 shuf = _mm_movehdup_ps(s128);
          __m128 sums = _mm_add_ps(s128, shuf);
          shuf = _mm_movehl_ps(shuf, sums);
          sums = _mm_add_ss(sums, shuf);
          return _mm_cvtss_f32(sums);
        };

        float d0 = hsum512(acc0);
        float d1 = hsum512(acc1);
        float d2 = hsum512(acc2);
        float d3 = hsum512(acc3);
        float d4 = hsum512(acc4);
        float d5 = hsum512(acc5);
        float d6 = hsum512(acc6);
        float d7 = hsum512(acc7);

        for (; k < vec_dim; ++k) {
          const float tv = tj[k];
          d0 += q0[k] * tv;
          d1 += q1[k] * tv;
          d2 += q2[k] * tv;
          d3 += q3[k] * tv;
          d4 += q4[k] * tv;
          d5 += q5[k] * tv;
          d6 += q6[k] * tv;
          d7 += q7[k] * tv;
        }

        if (d0 > max_dot[i + 0])
          max_dot[i + 0] = d0;
        if (d1 > max_dot[i + 1])
          max_dot[i + 1] = d1;
        if (d2 > max_dot[i + 2])
          max_dot[i + 2] = d2;
        if (d3 > max_dot[i + 3])
          max_dot[i + 3] = d3;
        if (d4 > max_dot[i + 4])
          max_dot[i + 4] = d4;
        if (d5 > max_dot[i + 5])
          max_dot[i + 5] = d5;
        if (d6 > max_dot[i + 6])
          max_dot[i + 6] = d6;
        if (d7 > max_dot[i + 7])
          max_dot[i + 7] = d7;
      }

      for (; i < count; ++i) {
        const float *qi = q + i * vec_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 15 < vec_dim; k += 16) {
          __m512 t = _mm512_loadu_ps(tj + k);
          __m512 v = _mm512_loadu_ps(qi + k);
          acc = _mm512_fmadd_ps(t, v, acc);
        }
        __m256 lo = _mm512_castps512_ps256(acc);
        __m256 hi = _mm512_extractf32x8_ps(acc, 1);
        __m256 s = _mm256_add_ps(lo, hi);
        __m128 lo128 = _mm256_castps256_ps128(s);
        __m128 hi128 = _mm256_extractf128_ps(s, 1);
        __m128 s128 = _mm_add_ps(lo128, hi128);
        __m128 shuf = _mm_movehdup_ps(s128);
        __m128 sums = _mm_add_ps(s128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        float d = _mm_cvtss_f32(_mm_add_ss(sums, shuf));
        for (; k < vec_dim; ++k)
          d += qi[k] * tj[k];
        if (d > max_dot[i])
          max_dot[i] = d;
      }

#else
      for (uint32_t i = 0; i < count; ++i) {
        const float *qi = q + i * vec_dim;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int k = 0;
        for (; k + 3 < vec_dim; k += 4) {
          a0 += qi[k + 0] * tj[k + 0];
          a1 += qi[k + 1] * tj[k + 1];
          a2 += qi[k + 2] * tj[k + 2];
          a3 += qi[k + 3] * tj[k + 3];
        }
        float d = (a0 + a1) + (a2 + a3);
        for (; k < vec_dim; ++k)
          d += qi[k] * tj[k];
        if (d > max_dot[i])
          max_dot[i] = d;
      }
#endif
    }

    float similarity = 0.0f;
    for (uint32_t i = 0; i < count; ++i)
      similarity += max_dot[i];

    if (max_dot != max_buf_static)
      delete[] max_dot;
    return similarity;
  }

  std::vector<float>
  distance_to_indices(const float *q, uint32_t count,
                      const std::vector<int> &indices) const {
    std::vector<float> distances;
    distances.reserve(indices.size());

    for (int idx : indices) {
      float sim = compute_chamfer_similarity(q, count, idx);
      distances.push_back(sim);
    }

    return distances;
  }

  std::vector<int> query_subset(const float *q, uint32_t count, int k,
                                const std::vector<int> &indices) const {
    std::vector<float> distances = distance_to_indices(q, count, indices);

    std::vector<std::pair<float, int>> similarities;
    similarities.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
      similarities.push_back({distances[i], indices[i]});
    }

    int subset_size = static_cast<int>(indices.size());
    std::partial_sort(
        similarities.begin(), similarities.begin() + std::min(k, subset_size),
        similarities.end(),
        [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
          return a.first > b.first;
        });

    std::vector<int> result;
    result.reserve(k);
    for (int i = 0; i < std::min(k, subset_size); i++) {
      result.push_back(similarities[i].second);
    }

    return result;
  }

  std::vector<int> query(const float *q, uint32_t count, int k) const {
    std::vector<int> all_indices(num_train_points);
    for (int i = 0; i < num_train_points; i++) {
      all_indices[i] = i;
    }
    return query_subset(q, count, k, all_indices);
  }

  std::vector<std::vector<int>> batch_query(const float *queries,
                                            const uint32_t *query_counts,
                                            int num_queries, int k) const {
    std::vector<std::vector<int>> results;
    results.reserve(num_queries);

    uint32_t offset = 0;
    for (int i = 0; i < num_queries; i++) {
      const float *q = queries + offset;
      uint32_t count = query_counts[i];
      results.push_back(query(q, count, k));
      offset += count * vec_dim;
    }

    return results;
  }

  std::vector<std::vector<int>> batch_query_subset(const float *queries,
                                                   const uint32_t *query_counts,
                                                   int num_queries, int k,
                                                   const int *indices_matrix,
                                                   int num_indices) const {
    std::vector<std::vector<int>> results;
    results.reserve(num_queries);

    uint32_t offset = 0;
    for (int i = 0; i < num_queries; i++) {
      const float *q = queries + offset;
      uint32_t count = query_counts[i];

      std::vector<int> query_indices(num_indices);
      for (int j = 0; j < num_indices; j++) {
        query_indices[j] = indices_matrix[i * num_indices + j];
      }

      results.push_back(query_subset(q, count, k, query_indices));
      offset += count * vec_dim;
    }

    return results;
  }

  std::vector<std::vector<float>>
  batch_distance_to_indices(const float *queries, const uint32_t *query_counts,
                            int num_queries, const int *indices_matrix,
                            int num_indices) const {
    std::vector<std::vector<float>> results;
    results.reserve(num_queries);

    uint32_t offset = 0;
    for (int i = 0; i < num_queries; i++) {
      const float *q = queries + offset;
      uint32_t count = query_counts[i];

      std::vector<int> query_indices(num_indices);
      for (int j = 0; j < num_indices; j++) {
        query_indices[j] = indices_matrix[i * num_indices + j];
      }

      results.push_back(distance_to_indices(q, count, query_indices));
      offset += count * vec_dim;
    }

    return results;
  }

  std::vector<std::vector<int>> batch_query_fixed(const float *queries,
                                                  int num_queries,
                                                  uint32_t query_vec_count,
                                                  int k) const {
    std::vector<std::vector<int>> results;
    results.reserve(num_queries);

    uint32_t stride = query_vec_count * vec_dim;
    for (int i = 0; i < num_queries; i++) {
      const float *q = queries + i * stride;
      results.push_back(query(q, query_vec_count, k));
    }

    return results;
  }

  std::vector<std::vector<int>>
  batch_query_subset_fixed(const float *queries, int num_queries,
                           uint32_t query_vec_count, int k,
                           const int *indices_matrix, int num_indices) const {
    std::vector<std::vector<int>> results;
    results.reserve(num_queries);

    uint32_t stride = query_vec_count * vec_dim;
    for (int i = 0; i < num_queries; i++) {
      const float *q = queries + i * stride;

      std::vector<int> query_indices(num_indices);
      for (int j = 0; j < num_indices; j++) {
        query_indices[j] = indices_matrix[i * num_indices + j];
      }

      results.push_back(query_subset(q, query_vec_count, k, query_indices));
    }

    return results;
  }

  std::vector<std::vector<float>> batch_distance_to_indices_fixed(
      const float *queries, int num_queries, uint32_t query_vec_count,
      const int *indices_matrix, int num_indices) const {
    std::vector<std::vector<float>> results;
    results.reserve(num_queries);

    uint32_t stride = query_vec_count * vec_dim;
    for (int i = 0; i < num_queries; i++) {
      const float *q = queries + i * stride;

      std::vector<int> query_indices(num_indices);
      for (int j = 0; j < num_indices; j++) {
        query_indices[j] = indices_matrix[i * num_indices + j];
      }

      results.push_back(distance_to_indices(q, query_vec_count, query_indices));
    }

    return results;
  }
};
