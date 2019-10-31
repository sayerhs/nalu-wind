/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/MomentumSynthTurbNodeKernel.h"
#include "ngp_utils/NgpFieldUtils.h"

namespace sierra {
namespace nalu {

void
MomentumSynthTurbNodeKernel::setup(Realm& realm)
{
  const auto& meshInfo = realm.mesh_info();
  density_ = nalu_ngp::get_ngp_field(meshInfo, "density");
  dualVol_ = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  turbForcing_ = nalu_ngp::get_ngp_field(meshInfo, "synth_turb_forcing");
}

void
MomentumSynthTurbNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const double fac = density_.get(node, 0) * dualVol_.get(node, 0);

  for (int i=0; i < NodeKernelTraits::NDimMax; ++i)
    rhs(i) += fac * turbForcing_.get(node, i);
}

}  // nalu
}  // sierra
