/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMSYNTHTURBNODEKERNEL_H
#define MOMENTUMSYNTHTURBNODEKERNEL_H

#include "node_kernels/NodeKernel.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;

class MomentumSynthTurbNodeKernel : public NGPNodeKernel<MomentumSynthTurbNodeKernel>
{
public:
  KOKKOS_FUNCTION
  MomentumSynthTurbNodeKernel() = default;

  KOKKOS_FUNCTION
  virtual ~MomentumSynthTurbNodeKernel() = default;

  virtual void setup(Realm&) override;

  KOKKOS_FUNCTION
  virtual void execute(
    NodeKernelTraits::LhsType&,
    NodeKernelTraits::RhsType&,
    const stk::mesh::FastMeshIndex&) override;

private:
  ngp::Field<double> density_;
  ngp::Field<double> dualVol_;
  ngp::Field<double> turbForcing_;

  unsigned densityID_ {stk::mesh::InvalidOrdinal};
  unsigned dualVolID_ {stk::mesh::InvalidOrdinal};
  unsigned synthTurbForcingID_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* MOMENTUMSYNTHTURBNODEKERNEL_H */
