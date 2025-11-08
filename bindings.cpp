#include "chamfer.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>

namespace py = pybind11;

PYBIND11_MODULE(chamfer, m) {
  m.doc() = "Chamfer distance-based nearest neighbor search";

  py::class_<Chamfer>(m, "Chamfer")
      .def(py::init([](py::array_t<float> train,
                       py::array_t<int32_t> train_counts) {
             py::buffer_info train_buf = train.request();
             py::buffer_info counts_buf = train_counts.request();

             if (train_buf.ndim != 2) {
               throw std::runtime_error("train must be a 2-dimensional array "
                                        "(total_vectors, vec_dim)");
             }
             if (counts_buf.ndim != 1) {
               throw std::runtime_error(
                   "train_counts must be a 1-dimensional array");
             }

             const float *train_ptr = static_cast<float *>(train_buf.ptr);
             const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
             int num_train_points = counts_buf.shape[0];
             int vec_dim = train_buf.shape[1];

             return new Chamfer(train_ptr, counts_ptr, vec_dim,
                                num_train_points);
           }),
           py::arg("train"), py::arg("train_counts"),
           R"doc(
            Initialize Chamfer index with training data.

            Parameters
            ----------
            train : numpy.ndarray
                2D array of training vectors (shape: [total_vectors, vec_dim])
            train_counts : numpy.ndarray
                Number of vectors per training point (shape: [num_train_points])
        )doc")

      .def(
          "compute_chamfer_similarity",
          [](const Chamfer &self, py::array_t<float> q, int train_point_idx) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 2) {
              throw std::runtime_error(
                  "query must be a 2-dimensional array (num_vectors, vec_dim)");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            int32_t count = q_buf.shape[0];
            return self.compute_chamfer_similarity(q_ptr, count,
                                                   train_point_idx);
          },
          py::arg("q"), py::arg("train_point_idx"),
          R"doc(
                Compute Chamfer similarity between a query and a specific training point.

                Parameters
                ----------
                q : numpy.ndarray
                    2D array of query vectors (shape: [num_vectors, vec_dim])
                train_point_idx : int
                    Index of the training point to compare against

                Returns
                -------
                float
                    Chamfer similarity score (higher is more similar)
            )doc")

      .def(
          "distance_to_indices",
          [](const Chamfer &self, py::array_t<float> q,
             const std::vector<int> &indices) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 2) {
              throw std::runtime_error(
                  "query must be a 2-dimensional array (num_vectors, vec_dim)");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            int32_t count = q_buf.shape[0];
            auto distances = self.distance_to_indices(q_ptr, count, indices);

            py::array_t<float> result_array(distances.size());
            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);
            for (size_t i = 0; i < distances.size(); ++i) {
              result_ptr[i] = distances[i];
            }

            return result_array;
          },
          py::arg("q"), py::arg("indices"),
          R"doc(
                Compute Chamfer distances to specific training point indices.

                Parameters
                ----------
                q : numpy.ndarray
                    2D array of query vectors (shape: [num_vectors, vec_dim])
                indices : list of int
                    Indices of training points to compute distances to

                Returns
                -------
                numpy.ndarray
                    Float32 array of Chamfer similarity scores in the same order as indices
            )doc")

      .def(
          "distance_to_points",
          [](const Chamfer &self, py::array_t<float> q) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 2) {
              throw std::runtime_error(
                  "query must be a 2-dimensional array (num_vectors, vec_dim)");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            int32_t count = q_buf.shape[0];
            auto distances = self.distance_to_points(q_ptr, count);

            py::array_t<float> result_array(distances.size());
            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);
            for (size_t i = 0; i < distances.size(); ++i) {
              result_ptr[i] = distances[i];
            }

            return result_array;
          },
          py::arg("q"),
          R"doc(
                Compute Chamfer distances to all training points.

                Parameters
                ----------
                q : numpy.ndarray
                    2D array of query vectors (shape: [num_vectors, vec_dim])

                Returns
                -------
                numpy.ndarray
                    Float32 array with Chamfer similarity scores for every training point
            )doc")

      .def(
          "batch_distance_to_points",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<int32_t> query_counts) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error(
                  "queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }

            bool queries_contiguous = queries_buf.strides[0] == sizeof(float);
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            bool counts_contiguous = counts_buf.strides[0] == sizeof(int32_t);
            if (!counts_contiguous) {
              throw std::runtime_error(
                  "query_counts array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
            int num_queries = counts_buf.shape[0];

            auto results = self.batch_distance_to_points(queries_ptr, counts_ptr,
                                                         num_queries);

            int num_points = self.train_point_count();

            py::array_t<float> result_array({num_queries, num_points});
            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              for (int j = 0; j < num_points; j++) {
                result_ptr[i * num_points + j] = results[i][j];
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("query_counts"),
          R"doc(
                Compute Chamfer distances to all training points for multiple queries (variable query sizes).

                Parameters
                ----------
                queries : numpy.ndarray
                    1D array containing all query vectors concatenated.
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])

                Returns
                -------
                numpy.ndarray
                    2D array of Chamfer similarity scores (shape: [num_queries, num_train_points])
            )doc")

      .def(
          "query_subset",
          [](const Chamfer &self, py::array_t<float> q, int k,
             const std::vector<int> &indices) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 2) {
              throw std::runtime_error(
                  "query must be a 2-dimensional array (num_vectors, vec_dim)");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            int32_t count = q_buf.shape[0];
            auto neighbors = self.query_subset(q_ptr, count, k, indices);

            py::array_t<int> result_array(neighbors.size());
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);
            for (size_t i = 0; i < neighbors.size(); ++i) {
              result_ptr[i] = neighbors[i];
            }

            return result_array;
          },
          py::arg("q"), py::arg("k"), py::arg("indices"),
          R"doc(
                Find k nearest neighbors from a subset of training points.

                Parameters
                ----------
                q : numpy.ndarray
                    2D array of query vectors (shape: [num_vectors, vec_dim])
                k : int
                    Number of nearest neighbors to return
                indices : list of int
                    Subset of training point indices to search

                Returns
                -------
                numpy.ndarray
                    Int32 array containing the indices of k nearest neighbors from the subset
            )doc")

      .def(
          "query",
          [](const Chamfer &self, py::array_t<float> q, int k) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 2) {
              throw std::runtime_error(
                  "query must be a 2-dimensional array (num_vectors, vec_dim)");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            int32_t count = q_buf.shape[0];
            auto neighbors = self.query(q_ptr, count, k);

            py::array_t<int> result_array(neighbors.size());
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);
            for (size_t i = 0; i < neighbors.size(); ++i) {
              result_ptr[i] = neighbors[i];
            }

            return result_array;
          },
          py::arg("q"), py::arg("k"),
          R"doc(
                Find k nearest neighbors from all training points.

                Parameters
                ----------
                q : numpy.ndarray
                    2D array of query vectors (shape: [num_vectors, vec_dim])
                k : int
                    Number of nearest neighbors to return

                Returns
                -------
                numpy.ndarray
                    Int32 array containing the indices of k nearest neighbors
            )doc")

      .def(
          "batch_query",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<int32_t> query_counts, int k) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error("queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
            int num_queries = counts_buf.shape[0];

            auto results =
                self.batch_query(queries_ptr, counts_ptr, num_queries, k);

            py::array_t<int> result_array({num_queries, k});
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              int actual_k = static_cast<int>(results[i].size());
              for (int j = 0; j < actual_k; j++) {
                result_ptr[i * k + j] = results[i][j];
              }
              for (int j = actual_k; j < k; j++) {
                result_ptr[i * k + j] = -1;
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("k"),
          R"doc(
                Find k nearest neighbors for multiple queries at once.

                Parameters
                ----------
                queries : numpy.ndarray
                    Flattened array of all query vectors (shape: [total_query_vectors * vec_dim])
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])
                k : int
                    Number of nearest neighbors to return per query

                Returns
                -------
                numpy.ndarray
                    Int32 array of shape (num_queries, k) containing nearest neighbor indices;
                    entries beyond available neighbors are filled with -1
            )doc")

      .def(
          "batch_query_subset",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<int32_t> query_counts, int k,
             py::array_t<int> indices) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();
            py::buffer_info indices_buf = indices.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error("queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }
            if (indices_buf.ndim != 2) {
              throw std::runtime_error("indices must be a 2-dimensional array "
                                       "(num_queries, num_indices)");
            }

            bool indices_contiguous =
                indices_buf.strides[1] == sizeof(int) &&
                indices_buf.strides[0] == indices_buf.shape[1] * sizeof(int);
            if (!indices_contiguous) {
              throw std::runtime_error(
                  "indices array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
            const int *indices_ptr = static_cast<int *>(indices_buf.ptr);
            int num_queries = counts_buf.shape[0];
            int num_indices = indices_buf.shape[1];

            auto results =
                self.batch_query_subset(queries_ptr, counts_ptr, num_queries, k,
                                        indices_ptr, num_indices);

            // Convert to 2D numpy array
            py::array_t<int> result_array({num_queries, k});
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              int actual_k = static_cast<int>(results[i].size());
              for (int j = 0; j < actual_k; j++) {
                result_ptr[i * k + j] = results[i][j];
              }
              for (int j = actual_k; j < k; j++) {
                result_ptr[i * k + j] = -1;
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("k"),
          py::arg("indices"),
          R"doc(
                Find k nearest neighbors from a subset for multiple queries.

                Parameters
                ----------
                queries : numpy.ndarray
                    Flattened array of all query vectors (shape: [total_query_vectors * vec_dim])
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])
                k : int
                    Number of nearest neighbors to return per query
                indices : numpy.ndarray
                    2D array of indices to search for each query (shape: [num_queries, num_indices])

                Returns
                -------
                numpy.ndarray
                    2D array of nearest neighbor indices (shape: [num_queries, k])
            )doc")

      .def(
          "batch_distance_to_indices",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<int32_t> query_counts, py::array_t<int> indices) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();
            py::buffer_info indices_buf = indices.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error("queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }
            if (indices_buf.ndim != 2) {
              throw std::runtime_error("indices must be a 2-dimensional array "
                                       "(num_queries, num_indices)");
            }

            bool indices_contiguous =
                indices_buf.strides[1] == sizeof(int) &&
                indices_buf.strides[0] == indices_buf.shape[1] * sizeof(int);
            if (!indices_contiguous) {
              throw std::runtime_error(
                  "indices array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
            const int *indices_ptr = static_cast<int *>(indices_buf.ptr);
            int num_queries = counts_buf.shape[0];
            int num_indices = indices_buf.shape[1];

            auto results = self.batch_distance_to_indices(
                queries_ptr, counts_ptr, num_queries, indices_ptr, num_indices);

            // Convert to 2D numpy array
            py::array_t<float> result_array({num_queries, num_indices});
            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              for (int j = 0; j < num_indices; j++) {
                result_ptr[i * num_indices + j] = results[i][j];
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("indices"),
          R"doc(
                Compute Chamfer distances to specific indices for multiple queries.

                Parameters
                ----------
                queries : numpy.ndarray
                    Flattened array of all query vectors (shape: [total_query_vectors * vec_dim])
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])
                indices : numpy.ndarray
                    2D array of indices to compute distances for each query (shape: [num_queries, num_indices])

                Returns
                -------
                numpy.ndarray
                    2D array of Chamfer similarity scores (shape: [num_queries, num_indices])
            )doc")

      .def(
          "batch_query_fixed",
          [](const Chamfer &self, py::array_t<float> queries, int k) {
            py::buffer_info queries_buf = queries.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }

            bool queries_contiguous =
                queries_buf.strides[2] == sizeof(float) &&
                queries_buf.strides[1] == queries_buf.shape[2] * sizeof(float) &&
                queries_buf.strides[0] == queries_buf.shape[1] * queries_buf.shape[2] * sizeof(float);
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];

            return self.batch_query_fixed(queries_ptr, num_queries,
                                          query_vec_count, k);
          },
          py::arg("queries"), py::arg("k"),
          R"doc(
                Find k nearest neighbors for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries have the same number of vectors.
                k : int
                    Number of nearest neighbors to return per query

                Returns
                -------
                list of list of int
                    For each query, indices of k nearest neighbors
            )doc")

      .def(
          "batch_query_subset_fixed",
          [](const Chamfer &self, py::array_t<float> queries, int k,
             py::array_t<int> indices) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info indices_buf = indices.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }
            if (indices_buf.ndim != 2) {
              throw std::runtime_error("indices must be a 2-dimensional array "
                                       "(num_queries, num_indices)");
            }

            bool queries_contiguous =
                queries_buf.strides[2] == sizeof(float) &&
                queries_buf.strides[1] == queries_buf.shape[2] * sizeof(float) &&
                queries_buf.strides[0] == queries_buf.shape[1] * queries_buf.shape[2] * sizeof(float);
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            bool indices_contiguous =
                indices_buf.strides[1] == sizeof(int) &&
                indices_buf.strides[0] == indices_buf.shape[1] * sizeof(int);
            if (!indices_contiguous) {
              throw std::runtime_error(
                  "indices array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int *indices_ptr = static_cast<int *>(indices_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];
            int num_indices = indices_buf.shape[1];

            auto results = self.batch_query_subset_fixed(
                queries_ptr, num_queries, query_vec_count, k, indices_ptr,
                num_indices);

            // Convert to 2D numpy array
            py::array_t<int> result_array({num_queries, k});
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              int actual_k = static_cast<int>(results[i].size());
              for (int j = 0; j < actual_k; j++) {
                result_ptr[i * k + j] = results[i][j];
              }
              for (int j = actual_k; j < k; j++) {
                result_ptr[i * k + j] = -1;
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("k"), py::arg("indices"),
          R"doc(
                Find k nearest neighbors from a subset for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries have the same number of vectors.
                k : int
                    Number of nearest neighbors to return per query
                indices : numpy.ndarray
                    2D array of indices to search for each query (shape: [num_queries, num_indices])

                Returns
                -------
                numpy.ndarray
                    2D array of nearest neighbor indices (shape: [num_queries, k])
            )doc")

      .def(
          "batch_distance_to_indices_fixed",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<int> indices) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info indices_buf = indices.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }
            if (indices_buf.ndim != 2) {
              throw std::runtime_error("indices must be a 2-dimensional array "
                                       "(num_queries, num_indices)");
            }

            bool queries_contiguous =
                queries_buf.strides[2] == sizeof(float) &&
                queries_buf.strides[1] == queries_buf.shape[2] * sizeof(float) &&
                queries_buf.strides[0] == queries_buf.shape[1] * queries_buf.shape[2] * sizeof(float);
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            bool indices_contiguous =
                indices_buf.strides[1] == sizeof(int) &&
                indices_buf.strides[0] == indices_buf.shape[1] * sizeof(int);
            if (!indices_contiguous) {
              throw std::runtime_error(
                  "indices array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int *indices_ptr = static_cast<int *>(indices_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];
            int num_indices = indices_buf.shape[1];

            auto results = self.batch_distance_to_indices_fixed(
                queries_ptr, num_queries, query_vec_count, indices_ptr,
                num_indices);

            // Convert to 2D numpy array
            py::array_t<float> result_array({num_queries, num_indices});
            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              for (int j = 0; j < num_indices; j++) {
                result_ptr[i * num_indices + j] = results[i][j];
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("indices"),
          R"doc(
                Compute Chamfer distances to specific indices for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries have the same number of vectors.
                indices : numpy.ndarray
                    2D array of indices to compute distances for each query (shape: [num_queries, num_indices])

                Returns
                -------
                numpy.ndarray
                    2D array of Chamfer similarity scores (shape: [num_queries, num_indices])
            )doc")

      .def(
          "batch_distance_to_points_fixed",
          [](const Chamfer &self, py::array_t<float> queries) {
            py::buffer_info queries_buf = queries.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }

            bool queries_contiguous =
                queries_buf.strides[2] == sizeof(float) &&
                queries_buf.strides[1] == queries_buf.shape[2] * sizeof(float) &&
                queries_buf.strides[0] ==
                    queries_buf.shape[1] * queries_buf.shape[2] * sizeof(float);
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];

            auto results = self.batch_distance_to_points_fixed(
                queries_ptr, num_queries, query_vec_count);

            int num_points = self.train_point_count();

            py::array_t<float> result_array({num_queries, num_points});
            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              for (int j = 0; j < num_points; j++) {
                result_ptr[i * num_points + j] = results[i][j];
              }
            }

            return result_array;
          },
          py::arg("queries"),
          R"doc(
                Compute Chamfer distances to all training points for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries share the same number of vectors.

                Returns
                -------
                numpy.ndarray
                    2D array of Chamfer similarity scores (shape: [num_queries, num_train_points])
            )doc")

      .def(
          "pairwise_similarities",
          [](const Chamfer &self) {
            auto similarities = self.pairwise_similarities();
            py::ssize_t num_points =
                static_cast<py::ssize_t>(similarities.size());

            std::vector<py::ssize_t> shape = {num_points, num_points};
            py::array_t<float> result_array(shape);

            auto result_buf = result_array.request();
            float *result_ptr = static_cast<float *>(result_buf.ptr);
            size_t row_length = static_cast<size_t>(num_points);

            for (py::ssize_t i = 0; i < num_points; ++i) {
              const auto &row = similarities[static_cast<size_t>(i)];
              if (row.size() != row_length) {
                throw std::runtime_error(
                    "pairwise similarities row size mismatch");
              }

              float *dest =
                  result_ptr + static_cast<size_t>(i) * row_length;
              std::copy(row.begin(), row.end(), dest);
            }

            return result_array;
          },
          R"doc(
                Compute the Chamfer similarities between every pair of training points.

                Returns
                -------
                numpy.ndarray
                    2D array of size (num_train_points, num_train_points) where entry [i, j]
                    equals chamfer(i -> j) / train_counts[i] + chamfer(j -> i) / train_counts[j].
            )doc");
}
