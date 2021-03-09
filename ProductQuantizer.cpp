/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "ProductQuantizer.h"

#include <cstddef>
#include <cstring>
#include <cstdio>
#include <memory>

#include <algorithm>

#include "FaissAssert.h"
#include "VectorTransform.h"
#include "IndexFlat.h"
#include "utils.h"

extern "C"
{

  /* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

  int sgemm_(const char *transa, const char *transb, FINTEGER *m, FINTEGER *n, FINTEGER *k, const float *alpha, const float *a,
             FINTEGER *lda, const float *b, FINTEGER *ldb, float *beta, float *c, FINTEGER *ldc);
}

namespace faiss
{

  /* compute an estimator using look-up tables for typical values of M */
  template <typename CT, class C>
  void pq_estimators_from_tables_Mmul4(int M, const CT *codes,
                                       size_t ncodes,
                                       const float *__restrict dis_table,
                                       size_t ksub,
                                       size_t k,
                                       float *heap_dis,
                                       long *heap_ids)
  {

    for (size_t j = 0; j < ncodes; j++)
    {
      float dis = 0;
      const float *dt = dis_table;

      for (size_t m = 0; m < M; m += 4)
      {
        float dism = 0;
        dism = dt[*codes++];
        dt += ksub;
        dism += dt[*codes++];
        dt += ksub;
        dism += dt[*codes++];
        dt += ksub;
        dism += dt[*codes++];
        dt += ksub;
        dis += dism;
      }

      if (C::cmp(heap_dis[0], dis))
      {
        heap_pop<C>(k, heap_dis, heap_ids);
        heap_push<C>(k, heap_dis, heap_ids, dis, j);
      }
    }
  }

  template <typename CT, class C>
  void pq_estimators_from_tables_M4(const CT *codes,
                                    size_t ncodes,
                                    const float *__restrict dis_table,
                                    size_t ksub,
                                    size_t k,
                                    float *heap_dis,
                                    long *heap_ids)
  {

    for (size_t j = 0; j < ncodes; j++)
    {
      float dis = 0;
      const float *dt = dis_table;
      dis = dt[*codes++];
      dt += ksub;
      dis += dt[*codes++];
      dt += ksub;
      dis += dt[*codes++];
      dt += ksub;
      dis += dt[*codes++];

      if (C::cmp(heap_dis[0], dis))
      {
        heap_pop<C>(k, heap_dis, heap_ids);
        heap_push<C>(k, heap_dis, heap_ids, dis, j);
      }
    }
  }

  template <typename CT, class C>
  static inline void pq_estimators_from_tables(const ProductQuantizer &pq,
                                               const CT *codes,
                                               size_t ncodes,
                                               const float *dis_table,
                                               size_t k,
                                               float *heap_dis,
                                               long *heap_ids)
  {

    if (pq.M == 4)
    {

      pq_estimators_from_tables_M4<CT, C>(codes, ncodes,
                                          dis_table, pq.ksub, k,
                                          heap_dis, heap_ids);
      return;
    }

    if (pq.M % 4 == 0)
    {
      pq_estimators_from_tables_Mmul4<CT, C>(pq.M, codes, ncodes,
                                             dis_table, pq.ksub, k,
                                             heap_dis, heap_ids);
      return;
    }

    /* Default is relatively slow */
    const size_t M = pq.M;
    const size_t ksub = pq.ksub;
    for (size_t j = 0; j < ncodes; j++)
    {
      float dis = 0;
      const float *__restrict dt = dis_table;
      for (int m = 0; m < M; m++)
      {
        dis += dt[*codes++];
        dt += ksub;
      }
      if (C::cmp(heap_dis[0], dis))
      {
        heap_pop<C>(k, heap_dis, heap_ids);
        heap_push<C>(k, heap_dis, heap_ids, dis, j);
      }
    }
  }

  template <class C>
  static inline void pq_estimators_from_tables_generic(const ProductQuantizer &pq,
                                                       size_t nbits,
                                                       const uint8_t *codes,
                                                       size_t ncodes,
                                                       const float *dis_table,
                                                       size_t k,
                                                       float *heap_dis,
                                                       long *heap_ids)
  {
    const size_t M = pq.M;
    const size_t ksub = pq.ksub;
    for (size_t j = 0; j < ncodes; ++j)
    {
      faiss::ProductQuantizer::PQDecoderGeneric decoder(
          codes + j * pq.code_size, nbits);
      float dis = 0;
      const float *__restrict dt = dis_table;
      for (size_t m = 0; m < M; m++)
      {
        uint64_t c = decoder.decode();
        dis += dt[c];
        dt += ksub;
      }

      if (C::cmp(heap_dis[0], dis))
      {
        heap_pop<C>(k, heap_dis, heap_ids);
        heap_push<C>(k, heap_dis, heap_ids, dis, j);
      }
    }
  }

  /*********************************************
 * PQ implementation
 *********************************************/

  ProductQuantizer::ProductQuantizer(size_t d, size_t M, size_t nbits) : d(d), M(M), nbits(nbits), assign_index(nullptr)
  {
    set_derived_values();
  }

  ProductQuantizer::ProductQuantizer()
      : ProductQuantizer(0, 1, 0) {}

  void ProductQuantizer::set_derived_values()
  {
    // quite a few derived values
    FAISS_THROW_IF_NOT(d % M == 0);
    dsub = d / M;
    code_size = (nbits * M + 7) / 8;
    ksub = 1 << nbits;
    centroids.resize(d * ksub);
    verbose = false;
    train_type = Train_default;
  }

  void ProductQuantizer::set_params(const float *centroids_, int m)
  {
    memcpy(get_centroids(m, 0), centroids_,
           ksub * dsub * sizeof(centroids_[0]));
  }

  static void init_hypercube(int d, int nbits,
                             int n, const float *x,
                             float *centroids)
  {

    std::vector<float> mean(d);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++)
        mean[j] += x[i * d + j];

    float maxm = 0;
    for (int j = 0; j < d; j++)
    {
      mean[j] /= n;
      if (fabs(mean[j]) > maxm)
        maxm = fabs(mean[j]);
    }

    for (int i = 0; i < (1 << nbits); i++)
    {
      float *cent = centroids + i * d;
      for (int j = 0; j < nbits; j++)
        cent[j] = mean[j] + (((i >> j) & 1) ? 1 : -1) * maxm;
      for (int j = nbits; j < d; j++)
        cent[j] = mean[j];
    }
  }

  static void init_hypercube_ksub(int d, int ksub,
                                  int n, const float *x,
                                  float *centroids)
  {

    std::vector<float> mean(d);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++)
        mean[j] += x[i * d + j];

    float maxm = 0;
    for (int j = 0; j < d; j++)
    {
      mean[j] /= n;
      if (fabs(mean[j]) > maxm)
        maxm = fabs(mean[j]);
    }

    for (int i = 0; i < ksub; i++)
    {
      float *cent = centroids + i * d;
      for (int j = 0; j < (int)floor(log2(ksub)); j++)
        cent[j] = mean[j] + (((i >> j) & 1) ? 1 : -1) * maxm;
      for (int j = (int)floor(log2(ksub)); j < d; j++)
        cent[j] = mean[j];
    }
  }

  static void init_hypercube_pca(int d, int nbits,
                                 int n, const float *x,
                                 float *centroids)
  {
    PCAMatrix pca(d, nbits);
    pca.train(n, x);

    for (int i = 0; i < (1 << nbits); i++)
    {
      float *cent = centroids + i * d;
      for (int j = 0; j < d; j++)
      {
        cent[j] = pca.mean[j];
        float f = 1.0;
        for (int k = 0; k < nbits; k++)
          cent[j] += f *
                     sqrt(pca.eigenvalues[k]) *
                     (((i >> k) & 1) ? 1 : -1) *
                     pca.PCAMat[j + k * d];
      }
    }
  }

  void ProductQuantizer::train(int n, const float *x)
  {
    if (train_type != Train_shared)
    {
      train_type_t final_train_type;
      final_train_type = train_type;
      if (train_type == Train_hypercube ||
          train_type == Train_hypercube_pca)
      {
        if (dsub < nbits)
        {
          final_train_type = Train_default;
          printf("cannot train hypercube: nbits=%ld > log2(d=%ld)\n",
                 nbits, dsub);
        }
      }

      float *xslice = new float[n * dsub];
      ScopeDeleter<float> del(xslice);
      for (int m = 0; m < M; m++)
      {
        for (int j = 0; j < n; j++)
          memcpy(xslice + j * dsub,
                 x + j * d + m * dsub,
                 dsub * sizeof(float));

        Clustering clus(dsub, ksub, cp);

        // we have some initialization for the centroids
        if (final_train_type != Train_default)
        {
          clus.centroids.resize(dsub * ksub);
        }

        switch (final_train_type)
        {
        case Train_hypercube:
          init_hypercube(dsub, nbits, n, xslice,
                         clus.centroids.data());
          break;
        case Train_hypercube_pca:
          init_hypercube_pca(dsub, nbits, n, xslice,
                             clus.centroids.data());
          break;
        case Train_hot_start:
          memcpy(clus.centroids.data(),
                 get_centroids(m, 0),
                 dsub * ksub * sizeof(float));
          break;
        default:;
        }

        if (verbose)
        {
          clus.verbose = true;
          printf("Training PQ slice %d/%zd\n", m, M);
        }
        IndexFlatL2 index(dsub);
        clus.train(n, xslice, assign_index ? *assign_index : index);
        set_params(clus.centroids.data(), m);
      }
    }
    else
    {

      Clustering clus(dsub, ksub, cp);

      if (verbose)
      {
        clus.verbose = true;
        printf("Training all PQ slices at once\n");
      }

      IndexFlatL2 index(dsub);

      clus.train(n * M, x, assign_index ? *assign_index : index);
      for (int m = 0; m < M; m++)
      {
        set_params(clus.centroids.data(), m);
      }
    }
  }

  template <class PQEncoder>
  void compute_code(const ProductQuantizer &pq, const float *x, uint8_t *code)
  {
    float distances[pq.ksub];
    PQEncoder encoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++)
    {
      float mindis = 1e20;
      uint64_t idxm = 0;
      const float *xsub = x + m * pq.dsub;

      fvec_L2sqr_ny(distances, xsub, pq.get_centroids(m, 0), pq.dsub, pq.ksub);

      /* Find best centroid */
      for (size_t i = 0; i < pq.ksub; i++)
      {
        float dis = distances[i];
        if (dis < mindis)
        {
          mindis = dis;
          idxm = i;
        }
      }

      encoder.encode(idxm);
    }
  }

  void ProductQuantizer::compute_code(const float *x, uint8_t *code) const
  {
    switch (nbits)
    {
    case 8:
      faiss::compute_code<PQEncoder8>(*this, x, code);
      break;

    case 16:
      faiss::compute_code<PQEncoder16>(*this, x, code);
      break;

    default:
      faiss::compute_code<PQEncoderGeneric>(*this, x, code);
      break;
    }
  }

  template <class PQDecoder>
  void decode(const ProductQuantizer &pq, const uint8_t *code, float *x)
  {
    PQDecoder decoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++)
    {
      uint64_t c = decoder.decode();
      memcpy(x + m * pq.dsub, pq.get_centroids(m, c), sizeof(float) * pq.dsub);
    }
  }

  void ProductQuantizer::decode(const uint8_t *code, float *x) const
  {
    switch (nbits)
    {
    case 8:
      faiss::decode<PQDecoder8>(*this, code, x);
      break;

    case 16:
      faiss::decode<PQDecoder16>(*this, code, x);
      break;

    default:
      faiss::decode<PQDecoderGeneric>(*this, code, x);
      break;
    }
  }

  void ProductQuantizer::decode(const uint8_t *code, float *x, size_t n) const
  {
    for (size_t i = 0; i < n; i++)
    {
      this->decode(code + code_size * i, x + d * i);
    }
  }

  void ProductQuantizer::compute_code_from_distance_table(const float *tab,
                                                          uint8_t *code) const
  {
    PQEncoderGeneric encoder(code, nbits);
    for (size_t m = 0; m < M; m++)
    {
      float mindis = 1e20;
      uint64_t idxm = 0;

      /* Find best centroid */
      for (size_t j = 0; j < ksub; j++)
      {
        float dis = *tab++;
        if (dis < mindis)
        {
          mindis = dis;
          idxm = j;
        }
      }

      encoder.encode(idxm);
    }
  }

  void ProductQuantizer::compute_codes_with_assign_index(
      const float *x,
      uint8_t *codes,
      size_t n)
  {
    FAISS_THROW_IF_NOT(assign_index && assign_index->d == dsub);

    for (size_t m = 0; m < M; m++)
    {
      assign_index->reset();
      assign_index->add(ksub, get_centroids(m, 0));
      size_t bs = 65536;
      float *xslice = new float[bs * dsub];
      ScopeDeleter<float> del(xslice);
      idx_t *assign = new idx_t[bs];
      ScopeDeleter<idx_t> del2(assign);

      for (size_t i0 = 0; i0 < n; i0 += bs)
      {
        size_t i1 = std::min(i0 + bs, n);

        for (size_t i = i0; i < i1; i++)
        {
          memcpy(xslice + (i - i0) * dsub,
                 x + i * d + m * dsub,
                 dsub * sizeof(float));
        }

        assign_index->assign(i1 - i0, xslice, assign);

        if (nbits == 8)
        {
          uint8_t *c = codes + code_size * i0 + m;
          for (size_t i = i0; i < i1; i++)
          {
            *c = assign[i - i0];
            c += M;
          }
        }
        else if (nbits == 16)
        {
          uint16_t *c = (uint16_t *)(codes + code_size * i0 + m * 2);
          for (size_t i = i0; i < i1; i++)
          {
            *c = assign[i - i0];
            c += M;
          }
        }
        else
        {
          for (size_t i = i0; i < i1; ++i)
          {
            uint8_t *c = codes + code_size * i + ((m * nbits) / 8);
            uint8_t offset = (m * nbits) % 8;
            uint64_t ass = assign[i - i0];

            PQEncoderGeneric encoder(c, nbits, offset);
            encoder.encode(ass);
          }
        }
      }
    }
  }

  void ProductQuantizer::compute_codes(const float *x,
                                       uint8_t *codes,
                                       size_t n) const
  {
    // process by blocks to avoid using too much RAM
    size_t bs = 256 * 1024;
    if (n > bs)
    {
      for (size_t i0 = 0; i0 < n; i0 += bs)
      {
        size_t i1 = std::min(i0 + bs, n);
        compute_codes(x + d * i0, codes + code_size * i0, i1 - i0);
      }
      return;
    }

    if (dsub < 16)
    { // simple direct computation

#pragma omp parallel for
      for (size_t i = 0; i < n; i++)
        compute_code(x + i * d, codes + i * code_size);
    }
    else
    { // worthwile to use BLAS
      float *dis_tables = new float[n * ksub * M];
      ScopeDeleter<float> del(dis_tables);
      compute_distance_tables(n, x, dis_tables);

#pragma omp parallel for
      for (size_t i = 0; i < n; i++)
      {
        uint8_t *code = codes + i * code_size;
        const float *tab = dis_tables + i * ksub * M;
        compute_code_from_distance_table(tab, code);
      }
    }
  }

  void ProductQuantizer::compute_distance_table(const float *x,
                                                float *dis_table) const
  {
    size_t m;

    for (m = 0; m < M; m++)
    {
      fvec_L2sqr_ny(dis_table + m * ksub,
                    x + m * dsub,
                    get_centroids(m, 0),
                    dsub,
                    ksub);
    }
  }

  void ProductQuantizer::compute_inner_prod_table(const float *x,
                                                  float *dis_table) const
  {
    size_t m;

    for (m = 0; m < M; m++)
    {
      fvec_inner_products_ny(dis_table + m * ksub,
                             x + m * dsub,
                             get_centroids(m, 0),
                             dsub,
                             ksub);
    }
  }

  void ProductQuantizer::compute_distance_tables(
      size_t nx,
      const float *x,
      float *dis_tables) const
  {

    if (dsub < 16)
    {

#pragma omp parallel for
      for (size_t i = 0; i < nx; i++)
      {
        compute_distance_table(x + i * d, dis_tables + i * ksub * M);
      }
    }
    else
    { // use BLAS

      for (int m = 0; m < M; m++)
      {
        pairwise_L2sqr(dsub,
                       nx, x + dsub * m,
                       ksub, centroids.data() + m * dsub * ksub,
                       dis_tables + ksub * m,
                       d, dsub, ksub * M);
      }
    }
  }

  void ProductQuantizer::compute_inner_prod_tables(
      size_t nx,
      const float *x,
      float *dis_tables) const
  {

    if (dsub < 16)
    {

#pragma omp parallel for
      for (size_t i = 0; i < nx; i++)
      {
        compute_inner_prod_table(x + i * d, dis_tables + i * ksub * M);
      }
    }
    else
    { // use BLAS

      // compute distance tables
      for (int m = 0; m < M; m++)
      {
        FINTEGER ldc = ksub * M, nxi = nx, ksubi = ksub,
                 dsubi = dsub, di = d;
        float one = 1.0, zero = 0;

        sgemm_("Transposed", "Not transposed",
               &ksubi, &nxi, &dsubi,
               &one, &centroids[m * dsub * ksub], &dsubi,
               x + dsub * m, &di,
               &zero, dis_tables + ksub * m, &ldc);
      }
    }
  }

  template <class C>
  static void pq_knn_search_with_tables(
      const ProductQuantizer &pq,
      size_t nbits,
      const float *dis_tables,
      const uint8_t *codes,
      const size_t ncodes,
      HeapArray<C> *res,
      bool init_finalize_heap)
  {
    size_t k = res->k, nx = res->nh;
    size_t ksub = pq.ksub, M = pq.M;

#pragma omp parallel for
    for (size_t i = 0; i < nx; i++)
    {
      /* query preparation for asymmetric search: compute look-up tables */
      const float *dis_table = dis_tables + i * ksub * M;

      /* Compute distances and keep smallest values */
      long *__restrict heap_ids = res->ids + i * k;
      float *__restrict heap_dis = res->val + i * k;

      if (init_finalize_heap)
      {
        heap_heapify<C>(k, heap_dis, heap_ids);
      }

      switch (nbits)
      {
      case 8:
        pq_estimators_from_tables<uint8_t, C>(pq,
                                              codes, ncodes,
                                              dis_table,
                                              k, heap_dis, heap_ids);
        break;

      case 16:
        pq_estimators_from_tables<uint16_t, C>(pq,
                                               (uint16_t *)codes, ncodes,
                                               dis_table,
                                               k, heap_dis, heap_ids);
        break;

      default:
        pq_estimators_from_tables_generic<C>(pq,
                                             nbits,
                                             codes, ncodes,
                                             dis_table,
                                             k, heap_dis, heap_ids);
        break;
      }

      if (init_finalize_heap)
      {
        heap_reorder<C>(k, heap_dis, heap_ids);
      }
    }
  }

  void ProductQuantizer::search(const float *__restrict x,
                                size_t nx,
                                const uint8_t *codes,
                                const size_t ncodes,
                                float_maxheap_array_t *res,
                                bool init_finalize_heap) const
  {
    FAISS_THROW_IF_NOT(nx == res->nh);
    std::unique_ptr<float[]> dis_tables(new float[nx * ksub * M]);
    compute_distance_tables(nx, x, dis_tables.get());

    pq_knn_search_with_tables<CMax<float, long>>(
        *this, nbits, dis_tables.get(), codes, ncodes, res, init_finalize_heap);
  }

  void ProductQuantizer::search_ip(const float *__restrict x,
                                   size_t nx,
                                   const uint8_t *codes,
                                   const size_t ncodes,
                                   float_minheap_array_t *res,
                                   bool init_finalize_heap) const
  {
    FAISS_THROW_IF_NOT(nx == res->nh);
    std::unique_ptr<float[]> dis_tables(new float[nx * ksub * M]);
    compute_inner_prod_tables(nx, x, dis_tables.get());

    pq_knn_search_with_tables<CMin<float, long>>(
        *this, nbits, dis_tables.get(), codes, ncodes, res, init_finalize_heap);
  }

  static float sqr(float x)
  {
    return x * x;
  }

  void ProductQuantizer::compute_sdc_table()
  {
    sdc_table.resize(M * ksub * ksub);

    for (int m = 0; m < M; m++)
    {

      const float *cents = centroids.data() + m * ksub * dsub;
      float *dis_tab = sdc_table.data() + m * ksub * ksub;

      // TODO optimize with BLAS
      for (int i = 0; i < ksub; i++)
      {
        const float *centi = cents + i * dsub;
        for (int j = 0; j < ksub; j++)
        {
          float accu = 0;
          const float *centj = cents + j * dsub;
          for (int k = 0; k < dsub; k++)
            accu += sqr(centi[k] - centj[k]);
          dis_tab[i + j * ksub] = accu;
        }
      }
    }
  }

  void ProductQuantizer::search_sdc(const uint8_t *qcodes,
                                    size_t nq,
                                    const uint8_t *bcodes,
                                    const size_t nb,
                                    float_maxheap_array_t *res,
                                    bool init_finalize_heap) const
  {
    FAISS_THROW_IF_NOT(sdc_table.size() == M * ksub * ksub);
    FAISS_THROW_IF_NOT(nbits == 8);
    size_t k = res->k;

#pragma omp parallel for
    for (size_t i = 0; i < nq; i++)
    {

      /* Compute distances and keep smallest values */
      long *heap_ids = res->ids + i * k;
      float *heap_dis = res->val + i * k;
      const uint8_t *qcode = qcodes + i * code_size;

      if (init_finalize_heap)
        maxheap_heapify(k, heap_dis, heap_ids);

      const uint8_t *bcode = bcodes;
      for (size_t j = 0; j < nb; j++)
      {
        float dis = 0;
        const float *tab = sdc_table.data();
        for (int m = 0; m < M; m++)
        {
          dis += tab[bcode[m] + qcode[m] * ksub];
          tab += ksub * ksub;
        }
        if (dis < heap_dis[0])
        {
          maxheap_pop(k, heap_dis, heap_ids);
          maxheap_push(k, heap_dis, heap_ids, dis, j);
        }
        bcode += code_size;
      }

      if (init_finalize_heap)
        maxheap_reorder(k, heap_dis, heap_ids);
    }
  }

  ProductQuantizer::PQEncoderGeneric::PQEncoderGeneric(uint8_t *code, int nbits,
                                                       uint8_t offset)
      : code(code), offset(offset), nbits(nbits), reg(0)
  {
    assert(nbits <= 64);
    if (offset > 0)
    {
      reg = (*code & ((1 << offset) - 1));
    }
  }

  void ProductQuantizer::PQEncoderGeneric::encode(uint64_t x)
  {
    reg |= (uint8_t)(x << offset);
    x >>= (8 - offset);
    if (offset + nbits >= 8)
    {
      *code++ = reg;

      for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i)
      {
        *code++ = (uint8_t)x;
        x >>= 8;
      }

      offset += nbits;
      offset &= 7;
      reg = (uint8_t)x;
    }
    else
    {
      offset += nbits;
    }
  }

  ProductQuantizer::PQEncoderGeneric::~PQEncoderGeneric()
  {
    if (offset > 0)
    {
      *code = reg;
    }
  }

  ProductQuantizer::PQEncoder8::PQEncoder8(uint8_t *code, int nbits)
      : code(code)
  {
    assert(8 == nbits);
  }

  void ProductQuantizer::PQEncoder8::encode(uint64_t x)
  {
    *code++ = (uint8_t)x;
  }

  ProductQuantizer::PQEncoder16::PQEncoder16(uint8_t *code, int nbits)
      : code((uint16_t *)code)
  {
    assert(16 == nbits);
  }

  void ProductQuantizer::PQEncoder16::encode(uint64_t x)
  {
    *code++ = (uint16_t)x;
  }

  ProductQuantizer::PQDecoderGeneric::PQDecoderGeneric(const uint8_t *code,
                                                       int nbits)
      : code(code),
        offset(0),
        nbits(nbits),
        mask((1ull << nbits) - 1),
        reg(0)
  {
    assert(nbits <= 64);
  }

  uint64_t ProductQuantizer::PQDecoderGeneric::decode()
  {
    if (offset == 0)
    {
      reg = *code;
    }
    uint64_t c = (reg >> offset);

    if (offset + nbits >= 8)
    {
      uint64_t e = 8 - offset;
      ++code;
      for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i)
      {
        c |= ((uint64_t)(*code++) << e);
        e += 8;
      }

      offset += nbits;
      offset &= 7;
      if (offset > 0)
      {
        reg = *code;
        c |= ((uint64_t)reg << e);
      }
    }
    else
    {
      offset += nbits;
    }

    return c & mask;
  }

  ProductQuantizer::PQDecoder8::PQDecoder8(const uint8_t *code, int nbits)
      : code(code)
  {
    assert(8 == nbits);
  }

  uint64_t ProductQuantizer::PQDecoder8::decode()
  {
    return (uint64_t)(*code++);
  }

  ProductQuantizer::PQDecoder16::PQDecoder16(const uint8_t *code, int nbits)
      : code((uint16_t *)code)
  {
    assert(16 == nbits);
  }

  uint64_t ProductQuantizer::PQDecoder16::decode()
  {
    return (uint64_t)(*code++);
  }

  /*********************************************
 * MultiPQ implementation
 *********************************************/

  MultiPQ::MultiPQ(size_t d, size_t M, size_t nbits_low, size_t nbits_high) : d(d), M(M), nbits_low(nbits_low), nbits_high(nbits_high)
  {
    set_derived_values();
  }

  void MultiPQ::set_derived_values()
  {
    FAISS_THROW_IF_NOT(d % M == 0);
    dsub = d / M;
    code_size_low = (nbits_low * M + 7) / 8;
    code_size_high = (nbits_high * M + 7) / 8;
    ksub_low = 1 << nbits_low;
    ksub_high = 1 << nbits_high;
    centroids.resize(d * ksub_high);
  }

  void MultiPQ::set_params_low(const float *centroids_, int m)
  {
    memcpy(get_centroids(m, 0), centroids_,
           ksub_low * dsub * sizeof(centroids_[0]));
  }

  void MultiPQ::set_params_high(const float *centroids_, int m)
  {
    memcpy(get_centroids(m, ksub_low), centroids_,
           (ksub_high - ksub_low) * dsub * sizeof(centroids_[0]));
  }

  void MultiPQ::train_low(idx_t n, const float *x)
  {
    float *xslice = new float[n * dsub];
    ScopeDeleter<float> del(xslice);
    for (int m = 0; m < M; m++)
    {
      for (int j = 0; j < n; j++)
        memcpy(xslice + j * dsub,
               x + j * d + m * dsub,
               dsub * sizeof(float));

      Clustering clus(dsub, ksub_low, cp);
      /*
      clus.centroids.resize(dsub * ksub_low);
      init_hypercube(dsub, nbits_low, n, xslice,
                   clus.centroids.data());
      */

      /*
      if (verbose)
      {
        clus.verbose = true;
        printf("Training MultiPQ low slice %d/%zd\n", m, M);
      }
      */
      IndexFlatL2 index(dsub);
      //clus.train(n, xslice, assign_index ? *assign_index : index);
      clus.train(n, xslice, index);
      set_params_low(clus.centroids.data(), m);
    }
  }

  void MultiPQ::train_high(idx_t n, const float *x)
  {
    float *xslice = new float[n * dsub];
    ScopeDeleter<float> del(xslice);
    for (int m = 0; m < M; m++)
    {
      for (int j = 0; j < n; j++)
        memcpy(xslice + j * dsub,
               x + j * d + m * dsub,
               dsub * sizeof(float));

      Clustering clus(dsub, ksub_high - ksub_low, cp);
      /*
      clus.centroids.resize(dsub * (ksub_high-ksub_low));
      init_hypercube_ksub(dsub, ksub_high-ksub_low, n, xslice,
                     clus.centroids.data());
      */

      /*
      if (verbose)
      {
        clus.verbose = true;
        printf("Training MultiPQ low slice %d/%zd\n", m, M);
      }
      */
      IndexFlatL2 index(dsub);
      //clus.train(n, xslice, assign_index ? *assign_index : index);
      clus.train(n, xslice, index);
      set_params_high(clus.centroids.data(), m);
    }
  }

  template <class PQEncoder>
  void compute_code_low(const MultiPQ &pq, const float *x, uint8_t *code)
  {
    float distances[pq.ksub_low];
    PQEncoder encoder(code, pq.nbits_low);
    for (size_t m = 0; m < pq.M; m++)
    {
      float mindis = 1e20;
      uint64_t idxm = 0;
      const float *xsub = x + m * pq.dsub;

      fvec_L2sqr_ny(distances, xsub, pq.get_centroids(m, 0), pq.dsub, pq.ksub_low);

      /* Find best centroid */
      for (size_t i = 0; i < pq.ksub_low; i++)
      {
        float dis = distances[i];
        if (dis < mindis)
        {
          mindis = dis;
          idxm = i;
        }
      }

      encoder.encode(idxm);
    }
  }

  template <class PQEncoder>
  void compute_code_high(const MultiPQ &pq, const float *x, uint8_t *code)
  {
    float distances[pq.ksub_high];
    PQEncoder encoder(code, pq.nbits_high);
    for (size_t m = 0; m < pq.M; m++)
    {
      float mindis = 1e20;
      uint64_t idxm = 0;
      const float *xsub = x + m * pq.dsub;

      fvec_L2sqr_ny(distances, xsub, pq.get_centroids(m, 0), pq.dsub, pq.ksub_high);

      /* Find best centroid */
      for (size_t i = 0; i < pq.ksub_high; i++)
      {
        float dis = distances[i];
        if (dis < mindis)
        {
          mindis = dis;
          idxm = i;
        }
      }

      encoder.encode(idxm);
    }
  }

  void MultiPQ::compute_code_low(const float *x, uint8_t *code) const
  {
    switch (nbits_low)
    {
    case 8:
      faiss::compute_code_low<PQEncoder8>(*this, x, code);
      break;

    case 16:
      faiss::compute_code_low<PQEncoder16>(*this, x, code);
      break;

    default:
      faiss::compute_code_low<PQEncoderGeneric>(*this, x, code);
      break;
    }
  }

  void MultiPQ::compute_code_high(const float *x, uint8_t *code) const
  {
    switch (nbits_high)
    {
    case 8:
      faiss::compute_code_high<PQEncoder8>(*this, x, code);
      break;

    case 16:
      faiss::compute_code_high<PQEncoder16>(*this, x, code);
      break;

    default:
      faiss::compute_code_high<PQEncoderGeneric>(*this, x, code);
      break;
    }
  }

  template <class PQDecoder>
  void decode_low(const MultiPQ &pq, const uint8_t *code, float *x)
  {
    PQDecoder decoder(code, pq.nbits_low);
    for (size_t m = 0; m < pq.M; m++)
    {
      uint64_t c = decoder.decode();
      memcpy(x + m * pq.dsub, pq.get_centroids(m, c), sizeof(float) * pq.dsub);
    }
  }

  template <class PQDecoder>
  void decode_high(const MultiPQ &pq, const uint8_t *code, float *x)
  {
    PQDecoder decoder(code, pq.nbits_high);
    for (size_t m = 0; m < pq.M; m++)
    {
      uint64_t c = decoder.decode();
      memcpy(x + m * pq.dsub, pq.get_centroids(m, c), sizeof(float) * pq.dsub);
    }
  }

  void MultiPQ::decode_low(const uint8_t *code, float *x) const
  {
    switch (nbits_low)
    {
    case 8:
      faiss::decode_low<PQDecoder8>(*this, code, x);
      break;

    case 16:
      faiss::decode_low<PQDecoder16>(*this, code, x);
      break;

    default:
      faiss::decode_low<PQDecoderGeneric>(*this, code, x);
      break;
    }
  }

  void MultiPQ::decode_high(const uint8_t *code, float *x) const
  {
    switch (nbits_high)
    {
    case 8:
      faiss::decode_high<PQDecoder8>(*this, code, x);
      break;

    case 16:
      faiss::decode_high<PQDecoder16>(*this, code, x);
      break;

    default:
      faiss::decode_high<PQDecoderGeneric>(*this, code, x);
      break;
    }
  }

  void MultiPQ::compute_codes_low(const float *x,
                                  uint8_t *codes,
                                  size_t n) const
  {
    // process by blocks to avoid using too much RAM
    size_t bs = 256 * 1024;
    if (n > bs)
    {
      for (size_t i0 = 0; i0 < n; i0 += bs)
      {
        size_t i1 = std::min(i0 + bs, n);
        compute_codes_low(x + d * i0, codes + code_size_low * i0, i1 - i0);
      }
      return;
    }

    if (dsub < 16)
    { // simple direct computation

#pragma omp parallel for
      for (size_t i = 0; i < n; i++)
        compute_code_low(x + i * d, codes + i * code_size_low);
    }
    else
    { // worthwile to use BLAS
      float *dis_tables = new float[n * ksub_high * M];
      ScopeDeleter<float> del(dis_tables);
      compute_distance_tables(n, x, dis_tables);

#pragma omp parallel for
      for (size_t i = 0; i < n; i++)
      {
        uint8_t *code = codes + i * code_size_low;
        const float *tab = dis_tables + i * ksub_high * M;
        compute_code_from_distance_table_low(tab, code);
      }
    }
  }

  void MultiPQ::compute_codes_high(const float *x,
                                   uint8_t *codes,
                                   size_t n) const
  {
    // process by blocks to avoid using too much RAM
    size_t bs = 256 * 1024;
    if (n > bs)
    {
      for (size_t i0 = 0; i0 < n; i0 += bs)
      {
        size_t i1 = std::min(i0 + bs, n);
        compute_codes_high(x + d * i0, codes + code_size_high * i0, i1 - i0);
      }
      return;
    }

    if (dsub < 16)
    { // simple direct computation

#pragma omp parallel for
      for (size_t i = 0; i < n; i++)
        compute_code_high(x + i * d, codes + i * code_size_high);
    }
    else
    { // worthwile to use BLAS
      float *dis_tables = new float[n * ksub_high * M];
      ScopeDeleter<float> del(dis_tables);
      compute_distance_tables(n, x, dis_tables);

#pragma omp parallel for
      for (size_t i = 0; i < n; i++)
      {
        uint8_t *code = codes + i * code_size_high;
        const float *tab = dis_tables + i * ksub_high * M;
        compute_code_from_distance_table_high(tab, code);
      }
    }
  }

  void MultiPQ::compute_code_from_distance_table_low(const float *tab,
                                                     uint8_t *code) const
  {
    PQEncoderGeneric encoder(code, nbits_low);
    for (size_t m = 0; m < M; m++)
    {
      float mindis = 1e20;
      uint64_t idxm = 0;

      /* Find best centroid */
      for (size_t j = 0; j < ksub_low; j++)
      {
        float dis = *tab++;
        if (dis < mindis)
        {
          mindis = dis;
          idxm = j;
        }
      }
      tab = tab + (ksub_high - ksub_low);

      encoder.encode(idxm);
    }
  }

  void MultiPQ::compute_code_from_distance_table_high(const float *tab,
                                                      uint8_t *code) const
  {
    ProductQuantizer::PQEncoderGeneric encoder(code, nbits_high);
    for (size_t m = 0; m < M; m++)
    {
      float mindis = 1e20;
      uint64_t idxm = 0;

      /* Find best centroid */
      for (size_t j = 0; j < ksub_high; j++)
      {
        float dis = *tab++;
        if (dis < mindis)
        {
          mindis = dis;
          idxm = j;
        }
      }

      encoder.encode(idxm);
    }
  }

  void MultiPQ::compute_distance_table(const float *x,
                                       float *dis_table) const
  {
    size_t m;

    for (m = 0; m < M; m++)
    {
      fvec_L2sqr_ny(dis_table + m * ksub_high,
                    x + m * dsub,
                    get_centroids(m, 0),
                    dsub,
                    ksub_high);
    }
  }

  void MultiPQ::compute_distance_tables(
      size_t nx,
      const float *x,
      float *dis_tables) const
  {

    if (dsub < 16)
    {

#pragma omp parallel for
      for (size_t i = 0; i < nx; i++)
      {
        compute_distance_table(x + i * d, dis_tables + i * ksub_high * M);
      }
    }
    else
    { // use BLAS

      for (int m = 0; m < M; m++)
      {
        pairwise_L2sqr(dsub,
                       nx, x + dsub * m,
                       ksub_high, centroids.data() + m * dsub * ksub_high,
                       dis_tables + ksub_high * m,
                       d, dsub, ksub_high * M);
      }
    }
  }

  using idx_t = Index::idx_t;
  template <class C>
  static inline void multipq_estimators_from_tables_generic(const MultiPQ &pq,
                                                            size_t nbits,
                                                            size_t nbits_high,
                                                            const uint8_t *codes,
                                                            size_t ncodes,
                                                            const uint8_t *codes_high,
                                                            size_t ncodes_high,
                                                            const std::map<idx_t, idx_t> &high_lookup,
                                                            const float *dis_table,
                                                            size_t k,
                                                            float *heap_dis,
                                                            long *heap_ids)
  {
    const size_t M = pq.M;
    const size_t ksub = pq.ksub_high;
    size_t last_idx = 0;

    for (auto it = high_lookup.begin(); it != high_lookup.end(); it++)
    {
      while (last_idx < it->first)
      {
        MultiPQ::PQDecoderGeneric decoder(
            codes + last_idx * ((nbits * M + 7) / 8), nbits);

        float dis = 0;
        const float *__restrict dt = dis_table;
        for (size_t m = 0; m < M; m++)
        {
          uint64_t c = decoder.decode();
          dis += dt[c];
          dt += ksub;
        }

        if (C::cmp(heap_dis[0], dis))
        {
          heap_pop<C>(k, heap_dis, heap_ids);
          heap_push<C>(k, heap_dis, heap_ids, dis, last_idx);
        }
        last_idx++;
      }

      MultiPQ::PQDecoderGeneric decoder(
          codes_high + it->second * ((nbits_high * M + 7) / 8), nbits_high);

      float dis = 0;
      const float *__restrict dt = dis_table;
      for (size_t m = 0; m < M; m++)
      {
        uint64_t c = decoder.decode();
        dis += dt[c];
        dt += ksub;
      }

      if (C::cmp(heap_dis[0], dis))
      {
        heap_pop<C>(k, heap_dis, heap_ids);
        heap_push<C>(k, heap_dis, heap_ids, dis, last_idx);
      }
      last_idx++;
    }

    for (size_t j = last_idx; j < ncodes; ++j)
    {
      MultiPQ::PQDecoderGeneric decoder(
          codes + j * ((nbits * M + 7) / 8), nbits);

      float dis = 0;
      const float *__restrict dt = dis_table;
      for (size_t m = 0; m < M; m++)
      {
        uint64_t c = decoder.decode();
        dis += dt[c];
        dt += ksub;
      }

      if (C::cmp(heap_dis[0], dis))
      {
        heap_pop<C>(k, heap_dis, heap_ids);
        heap_push<C>(k, heap_dis, heap_ids, dis, j);
      }
    }
  }

  template <class C>
  static void multipq_knn_search_with_tables(
      const MultiPQ &mq,
      size_t nbits,
      size_t nbits_high,
      const float *dis_tables,
      const uint8_t *codes,
      const size_t ncodes,
      const uint8_t *codes_high,
      const size_t ncodes_high,
      const std::map<idx_t, idx_t> &high_lookup,
      HeapArray<C> *res,
      bool init_finalize_heap)
  {
    size_t k = res->k, nx = res->nh;
    size_t ksub = mq.ksub_high, M = mq.M;

#pragma omp parallel for
    for (size_t i = 0; i < nx; i++)
    {
      /* query preparation for asymmetric search: compute look-up tables */
      const float *dis_table = dis_tables + i * ksub * M;

      /* Compute distances and keep smallest values */
      long *__restrict heap_ids = res->ids + i * k;
      float *__restrict heap_dis = res->val + i * k;

      if (init_finalize_heap)
      {
        heap_heapify<C>(k, heap_dis, heap_ids);
      }

      switch (nbits)
      {
        /*
      case 8:
        multipq_estimators_from_tables<uint8_t, C>(mq,
                                              codes, ncodes,
                                              dis_table,
                                              k, heap_dis, heap_ids);
        break;

      case 16:
        multipq_estimators_from_tables<uint16_t, C>(mq,
                                               (uint16_t *)codes, ncodes,
                                               dis_table,
                                               k, heap_dis, heap_ids);
        break;
        */
      default:
        multipq_estimators_from_tables_generic<C>(mq,
                                                  nbits, nbits_high,
                                                  codes, ncodes,
                                                  codes_high, ncodes_high, high_lookup,
                                                  dis_table,
                                                  k, heap_dis, heap_ids);
        break;
      }

      if (init_finalize_heap)
      {
        heap_reorder<C>(k, heap_dis, heap_ids);
      }
    }
  }

  void MultiPQ::search(const float *x,
                       size_t nx,
                       const uint8_t *codes_low,
                       const uint8_t *codes_high,
                       const size_t ncodes_low,
                       const size_t ncodes_high,
                       const std::map<idx_t, idx_t> &high_lookup,
                       const std::vector<idx_t> &high_indexes,
                       float_maxheap_array_t *res,
                       bool init_finalize_heap) const
  {
    FAISS_THROW_IF_NOT(nx == res->nh);
    std::unique_ptr<float[]> dis_tables(new float[nx * ksub_high * M]);
    compute_distance_tables(nx, x, dis_tables.get());
    size_t k = res->k;

    //idx_t *labels_high = new idx_t[res->nh * res->k];
    //float *distances_high = new float[res->nh * res->k];
    //float_maxheap_array_t res_high{res->nh, res->k, labels_high, distances_high};
    multipq_knn_search_with_tables<CMax<float, long>>(
        *this, nbits_low, nbits_high, dis_tables.get(), codes_low, ncodes_low, codes_high, ncodes_high, high_lookup, res, init_finalize_heap);
    /*_knn_search_with_tables<CMax<float, long>>(
	 *this, nbits_high, dis_tables.get(), codes_high, ncodes_high, &res_high, init_finalize_heap);
    
    for (int i = 0; i < nx; i++)
    {
      using C = CMax<float, long>;
      long *__restrict heap_ids_high = res_high.ids + i * k;
      float *__restrict heap_dis_high = res_high.val + i * k;

      long *__restrict heap_ids = res->ids + i * k;
      float *__restrict heap_dis = res->val + i * k;
      for (int j = 0; j < k; j++)
      {
        if (C::cmp(heap_dis[0], heap_dis_high[j]))
        {
          heap_pop<C>(k, heap_dis, heap_ids);
          heap_push<C>(k, heap_dis, heap_ids, heap_dis_high[j], high_indexes[heap_ids_high[j]]);
        }
      }

      heap_reorder<C>(k, heap_dis, heap_ids);
    }
    
    delete labels_high, distances_high; 
	*/
  }

  MultiPQ::PQEncoderGeneric::PQEncoderGeneric(uint8_t *code, int nbits,
                                              uint8_t offset)
      : code(code), offset(offset), nbits(nbits), reg(0)
  {
    assert(nbits <= 64);
    if (offset > 0)
    {
      reg = (*code & ((1 << offset) - 1));
    }
  }

  void MultiPQ::PQEncoderGeneric::encode(uint64_t x)
  {
    reg |= (uint8_t)(x << offset);
    x >>= (8 - offset);
    if (offset + nbits >= 8)
    {
      *code++ = reg;

      for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i)
      {
        *code++ = (uint8_t)x;
        x >>= 8;
      }

      offset += nbits;
      offset &= 7;
      reg = (uint8_t)x;
    }
    else
    {
      offset += nbits;
    }
  }

  MultiPQ::PQEncoderGeneric::~PQEncoderGeneric()
  {
    if (offset > 0)
    {
      *code = reg;
    }
  }

  MultiPQ::PQEncoder8::PQEncoder8(uint8_t *code, int nbits)
      : code(code)
  {
    assert(8 == nbits);
  }

  void MultiPQ::PQEncoder8::encode(uint64_t x)
  {
    *code++ = (uint8_t)x;
  }

  MultiPQ::PQEncoder16::PQEncoder16(uint8_t *code, int nbits)
      : code((uint16_t *)code)
  {
    assert(16 == nbits);
  }

  void MultiPQ::PQEncoder16::encode(uint64_t x)
  {
    *code++ = (uint16_t)x;
  }

  MultiPQ::PQDecoderGeneric::PQDecoderGeneric(const uint8_t *code,
                                              int nbits)
      : code(code),
        offset(0),
        nbits(nbits),
        mask((1ull << nbits) - 1),
        reg(0)
  {
    assert(nbits <= 64);
  }

  uint64_t MultiPQ::PQDecoderGeneric::decode()
  {
    if (offset == 0)
    {
      reg = *code;
    }
    uint64_t c = (reg >> offset);

    if (offset + nbits >= 8)
    {
      uint64_t e = 8 - offset;
      ++code;
      for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i)
      {
        c |= ((uint64_t)(*code++) << e);
        e += 8;
      }

      offset += nbits;
      offset &= 7;
      if (offset > 0)
      {
        reg = *code;
        c |= ((uint64_t)reg << e);
      }
    }
    else
    {
      offset += nbits;
    }

    return c & mask;
  }

  MultiPQ::PQDecoder8::PQDecoder8(const uint8_t *code, int nbits)
      : code(code)
  {
    assert(8 == nbits);
  }

  uint64_t MultiPQ::PQDecoder8::decode()
  {
    return (uint64_t)(*code++);
  }

  MultiPQ::PQDecoder16::PQDecoder16(const uint8_t *code, int nbits)
      : code((uint16_t *)code)
  {
    assert(16 == nbits);
  }

  uint64_t MultiPQ::PQDecoder16::decode()
  {
    return (uint64_t)(*code++);
  }
} // namespace faiss
