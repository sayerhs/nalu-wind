/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <algorithm>

#include "wind_energy/SyntheticTurbulence.h"
#include "NaluEnv.h"
#include "Realm.h"
#include "master_element/TensorOps.h"

#include "netcdf.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"

namespace sierra {
namespace nalu {
namespace synth_turb {

class LinearShearProfile : public MeanProfile
{
public:
  LinearShearProfile(double refVel, double refHeight, double slope, double height)
    : MeanProfile(refVel, refHeight),
      slope_(slope),
      halfHeight_(0.5 * height)
  {}

  virtual ~LinearShearProfile() = default;

  virtual double operator()(double ht) const
  {
    const double relHt = ht - refHeight_;
    if (relHt < -halfHeight_)
      return refVel_ * (1.0 - slope_ * halfHeight_);
    else if (relHt > halfHeight_)
      return refVel_ * (1.0 + slope_ * halfHeight_);
    else
      return refVel_ * (1.0 + slope_ * relHt);
  }

private:
  double slope_;
  double halfHeight_;
};

class PowerLawProfile : public MeanProfile
{
public:
  PowerLawProfile(double refVel, double refHeight, double alpha)
    : MeanProfile(refVel, refHeight),
      alpha_(alpha)
  {}

  virtual ~PowerLawProfile() = default;

  virtual double operator()(double height) const override
  {
    return refVel_ * std::pow((height / refHeight_), alpha_);
  }

private:
  const double alpha_;
};

} // namespace synth_turb

namespace {
/** Check NetCDF errors and throw runtime errors with a message
 */
inline void check_nc_error(int ierr)
{
  if (ierr != NC_NOERR)
    throw std::runtime_error(
      "SyntheticTurbulence NetCDF Error: " + std::string(nc_strerror(ierr)));
}

/** Parse the NetCDF turbulence database and determine details of the turbulence box.
 *
 *. Initializes the dimensions and grid length, sizes in SynthTurbData. Also
 *  allocates the necessary memory for the perturbation velocities.
 *
 *. @param turbFile Information regarding NetCDF data identifiers
 *. @param turbGrid Turbulence data
 */
void process_nc_file(
  SyntheticTurbulence::NCBoxTurb& turbFile,
  SynthTurbData& turbGrid)
{
  check_nc_error(nc_open(turbFile.filename.c_str(), NC_NOWRITE, &turbFile.ncid));

  size_t ndim, nx, ny, nz;
  check_nc_error(nc_inq_dimid(turbFile.ncid, "ndim", &turbFile.sDim));
  check_nc_error(nc_inq_dimlen(turbFile.ncid, turbFile.sDim, &ndim));
  ThrowRequire(ndim == SynthTurbTraits::NDimMax);

  // Grid dimensions
  check_nc_error(nc_inq_dimid(turbFile.ncid, "nx", &turbFile.xDim));
  check_nc_error(nc_inq_dimlen(turbFile.ncid, turbFile.xDim, &nx));
  check_nc_error(nc_inq_dimid(turbFile.ncid, "ny", &turbFile.yDim));
  check_nc_error(nc_inq_dimlen(turbFile.ncid, turbFile.yDim, &ny));
  check_nc_error(nc_inq_dimid(turbFile.ncid, "nz", &turbFile.zDim));
  check_nc_error(nc_inq_dimlen(turbFile.ncid, turbFile.zDim, &nz));

  turbGrid.boxDims_[0] = nx;
  turbGrid.boxDims_[1] = ny;
  turbGrid.boxDims_[2] = nz;

  // Box lengths and resolution
  check_nc_error(nc_inq_varid(turbFile.ncid, "box_lengths", &turbFile.boxLenid));
  check_nc_error(nc_get_var_double(turbFile.ncid, turbFile.boxLenid, turbGrid.boxLen_));
  check_nc_error(nc_inq_varid(turbFile.ncid, "dx", &turbFile.dxid));
  check_nc_error(nc_get_var_double(turbFile.ncid, turbFile.dxid, turbGrid.dx_));

  // Perturbation velocity info
  check_nc_error(nc_inq_varid(turbFile.ncid, "uvel", &turbFile.uid));
  check_nc_error(nc_inq_varid(turbFile.ncid, "vvel", &turbFile.vid));
  check_nc_error(nc_inq_varid(turbFile.ncid, "wvel", &turbFile.wid));
  nc_close(turbFile.ncid);

  // Create data structures to store the perturbation velocities for two planes
  // [t, t+dt] such that the time of interest is within this interval.
  // turbGrid.uvel_ = SynthTurbTraits::StructField("SynthTurbData::uvel", 2*ny*nz);
  // turbGrid.vvel_ = SynthTurbTraits::StructField("SynthTurbData::vvel", 2*ny*nz);
  // turbGrid.wvel_ = SynthTurbTraits::StructField("SynthTurbData::wvel", 2*ny*nz);
  // turbGrid.h_uvel_ = Kokkos::create_mirror_view(turbGrid.uvel_);
  // turbGrid.h_vvel_ = Kokkos::create_mirror_view(turbGrid.vvel_);
  // turbGrid.h_wvel_ = Kokkos::create_mirror_view(turbGrid.wvel_);
  const size_t gridSize = 2 * ny * nz;
  turbGrid.uvel_.resize(gridSize);
  turbGrid.vvel_.resize(gridSize);
  turbGrid.wvel_.resize(gridSize);
}

/** Load two planes of data that bound the current timestep
 *
 *  The data for the y and z directions are loaded for the entire grid at the two planes
 */
void load_turb_plane_data(
  SyntheticTurbulence::NCBoxTurb& turbFile,
  SynthTurbData& turbGrid,
  const int il, const int ir)
{
  check_nc_error(nc_open(turbFile.filename.c_str(), NC_NOWRITE, &turbFile.ncid));

  size_t start[SynthTurbTraits::NDimMax]{static_cast<size_t>(il), 0, 0};
  size_t count[SynthTurbTraits::NDimMax]{
    2, static_cast<size_t>(turbGrid.boxDims_[1]),
    static_cast<size_t>(turbGrid.boxDims_[2])};
  if ((ir - il) == 1) {
    // two consequtive planes load them in one shot
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.uid, start, count, &turbGrid.uvel_[0]));
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.vid, start, count, &turbGrid.vvel_[0]));
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.wid, start, count, &turbGrid.wvel_[0]));
  } else {
    // Load the planes separately
    count[0] = 1;
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.uid, start, count, &turbGrid.uvel_[0]));
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.vid, start, count, &turbGrid.vvel_[0]));
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.wid, start, count, &turbGrid.wvel_[0]));

    start[0] = static_cast<size_t>(ir);
    const size_t offset = turbGrid.boxDims_[1] * turbGrid.boxDims_[2];
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.uid, start, count, &turbGrid.uvel_[offset]));
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.vid, start, count, &turbGrid.vvel_[offset]));
    check_nc_error(nc_get_vara_double(
      turbFile.ncid, turbFile.wid, start, count, &turbGrid.wvel_[offset]));
  }

  // Update left and right indices for future checks
  turbGrid.iLeft_ = il;
  turbGrid.iRight_ = ir;

  nc_close(turbFile.ncid);
}

/** Transform a position vector from global inertial reference frame to local
 *  reference frame attached to the turbulence grid.
 */
void global_to_local(const SynthTurbData& turbGrid, const double* inp, double* out)
{
  const auto* trMat = turbGrid.trMat_;
  double in[SynthTurbTraits::NDimMax];
  for (int i=0; i < SynthTurbTraits::NDimMax; ++i)
    in[i] = inp[i] - turbGrid.origin_[i];

  out[0] = trMat[0][0] * in[0] + trMat[0][1] * in[1] + trMat[0][2] * in[2];
  out[1] = trMat[1][0] * in[0] + trMat[1][1] * in[1] + trMat[1][2] * in[2];
  out[2] = trMat[2][0] * in[0] + trMat[2][1] * in[1] + trMat[2][2] * in[2];
}

/** Transform a vector from local reference frame back to the global inertial frame
 *
 */
void local_to_global_vel(const SynthTurbData& turbGrid, const double* in, double* out)
{
  const auto* trMat = turbGrid.trMat_;
  out[0] = trMat[0][0] * in[0] + trMat[1][0] * in[1] + trMat[2][0] * in[2];
  out[1] = trMat[0][1] * in[0] + trMat[1][1] * in[1] + trMat[2][1] * in[2];
  out[2] = trMat[0][2] * in[0] + trMat[1][2] * in[1] + trMat[2][2] * in[2];
}

/** Determine the left/right indices for a given point along a particular direction
 *
 *  @param turbGrid Turbulence box data
 *  @param dir Direction of search (0 = x, 1 = y, 2 = z)
 *  @param xin Coordinate value in local coordinate frame corresponding to direction provided
 *  @param il Index of the lower bound (populated by this function)
 *  @param ir Index of the upper bound (populated by this function)
 */
void get_lr_indices(
  const SynthTurbData& turbGrid,
  const int dir,
  const double xin,
  int& il,
  int& ir)
{
  const double xbox = xin - std::floor(xin / turbGrid.boxLen_[dir]) * turbGrid.boxLen_[dir];

  il = static_cast<int>(std::floor(xbox / turbGrid.dx_[dir]));
  ir = il + 1;
  if (ir >= turbGrid.boxDims_[dir])
    ir -= turbGrid.boxDims_[dir];
}

/** Determine the left/right indices for a given point along a particular direction
 *
 *  This overload also populates the fractions of left/right states to be used
 *  for interpolations.
 *
 *  @param turbGrid Turbulence box data
 *  @param dir Direction of search (0 = x, 1 = y, 2 = z)
 *  @param xin Coordinate value in local coordinate frame corresponding to direction provided
 *  @param il Index of the lower bound (populated by this function)
 *  @param ir Index of the upper bound (populated by this function)
 */
void get_lr_indices(
  const SynthTurbData& turbGrid,
  const int dir,
  const double xin,
  int& il, int& ir,
  double& rxl, double& rxr)
{
  const double xbox = xin - std::floor(xin / turbGrid.boxLen_[dir]) * turbGrid.boxLen_[dir];

  il = static_cast<int>(std::floor(xbox / turbGrid.dx_[dir]));
  ir = il + 1;
  if (ir >= turbGrid.boxDims_[dir])
    ir -= turbGrid.boxDims_[dir];

  const double xFrac = xbox - turbGrid.dx_[dir] * il;
  rxl = xFrac / turbGrid.dx_[dir];
  rxr = (1.0 - rxl);
}

/** Indices and interpolation weights for a given point located within the
 *  turbulence box
 */
struct InterpWeights
{
  int il, ir, jl, jr, kl, kr;
  double xl, xr, yl, yr, zl, zr;
};

/** Determine if a given point (in local frame) is within the turbulence box
 *
 *  If the point is found within the box, also determine the indices and
 *  interpolation weights for the y and z directions.
 *
 *  @return True if the point is inside the 2-D box
 */
bool find_point_in_box(
  const SynthTurbData& tGrid,
  const double* pt,
  InterpWeights& wt)
{
  // Get y and z w.r.t. the lower corner of the grid
  const double yy = pt[1] + tGrid.boxLen_[1] * 0.5;
  const double zz = pt[2] + tGrid.boxLen_[2] * 0.5;

  bool inBox =
    ((yy >= 0.0) &&
     (yy <= tGrid.boxLen_[1]) &&
     (zz >= 0.0) &&
     (zz <= tGrid.boxLen_[2]));

  if (inBox) {
    get_lr_indices(tGrid, 1, yy, wt.jl, wt.jr, wt.yl, wt.yr);
    get_lr_indices(tGrid, 2, zz, wt.kl, wt.kr, wt.zl, wt.zr);
  }

  return inBox;
}

/** Interpolate the perturbation velocity to a given point from the grid data
 */
void interp_perturb_vel(
  const SynthTurbData& tGrid,
  const InterpWeights& wt,
  double* vel)
{
  const int nz = tGrid.boxDims_[2];
  const int nynz = tGrid.boxDims_[1] * tGrid.boxDims_[2];
  // Indices of the 2-D cell that contains the sampling point
  int qidx[4]{wt.jl * nz + wt.kl,
      wt.jr * nz + wt.kl,
      wt.jr * nz + wt.kr,
      wt.jl * nz + wt.kl};

  double velL[SynthTurbTraits::NDimMax], velR[SynthTurbTraits::NDimMax];

  // Left quad (t = t)
  velL[0] =
    wt.yl * wt.zl * tGrid.uvel_[qidx[0]] + wt.yr * wt.zl * tGrid.uvel_[qidx[1]] +
    wt.yr * wt.zr * tGrid.uvel_[qidx[2]] + wt.yl * wt.zr * tGrid.uvel_[qidx[3]];
  velL[1] =
    wt.yl * wt.zl * tGrid.vvel_[qidx[0]] + wt.yr * wt.zl * tGrid.vvel_[qidx[1]] +
    wt.yr * wt.zr * tGrid.vvel_[qidx[2]] + wt.yl * wt.zr * tGrid.vvel_[qidx[3]];
  velL[2] =
    wt.yl * wt.zl * tGrid.wvel_[qidx[0]] + wt.yr * wt.zl * tGrid.wvel_[qidx[1]] +
    wt.yr * wt.zr * tGrid.wvel_[qidx[2]] + wt.yl * wt.zr * tGrid.wvel_[qidx[3]];

  for (int i=0; i < 4; ++i)
    qidx[i] += nynz;

  // Right quad (t = t+deltaT)
  velR[0] =
    wt.yl * wt.zl * tGrid.uvel_[qidx[0]] + wt.yr * wt.zl * tGrid.uvel_[qidx[1]] +
    wt.yr * wt.zr * tGrid.uvel_[qidx[2]] + wt.yl * wt.zr * tGrid.uvel_[qidx[3]];
  velR[1] =
    wt.yl * wt.zl * tGrid.vvel_[qidx[0]] + wt.yr * wt.zl * tGrid.vvel_[qidx[1]] +
    wt.yr * wt.zr * tGrid.vvel_[qidx[2]] + wt.yl * wt.zr * tGrid.vvel_[qidx[3]];
  velR[2] =
    wt.yl * wt.zl * tGrid.wvel_[qidx[0]] + wt.yr * wt.zl * tGrid.wvel_[qidx[1]] +
    wt.yr * wt.zr * tGrid.wvel_[qidx[2]] + wt.yl * wt.zr * tGrid.wvel_[qidx[3]];

  // Interpolation in time
  for (int i=0; i < SynthTurbTraits::NDimMax; ++i)
    vel[i] = wt.xl * velL[i] + wt.xr * velR[i];
}

}

SyntheticTurbulence::SyntheticTurbulence(
  Realm& realm
) : realm_(realm)
{}

void SyntheticTurbulence::load(const YAML::Node& node)
{
  const double pi = std::acos(-1.0);
  // NetCDF file containing the turbulence data
  turbFile_.filename = node["turbulence_file"].as<std::string>();
  process_nc_file(turbFile_, turbGrid_);

  // Load position and orientation of the grid
  auto wind_direction = node["wind_direction"].as<double>();
  // Convert to radians
  wind_direction *= pi / 180.0;
  const auto location = node["grid_location"].as<std::vector<double>>();
  ThrowRequire(location.size() == 3u);

  std::string mean_wind_type = "uniform";
  double wind_speed;
  // Default reference height is the center of the turbulence grid
  double ref_height = location[2];
  get_required(node, "mean_wind_speed", wind_speed);
  get_if_present(node, "mean_wind_type", mean_wind_type, mean_wind_type);
  get_if_present(node, "mean_wind_ref_height", ref_height, ref_height);

  if (mean_wind_type == "constant") {
    windProfile_.reset(new synth_turb::MeanProfile(wind_speed, ref_height));
  } else if (mean_wind_type == "linear_shear") {
    const double shear_slope = node["shear_slope"].as<double>();
    const double shear_width = node["shear_width"].as<double>();
    windProfile_.reset(new synth_turb::LinearShearProfile(
      wind_speed, ref_height, shear_slope, shear_width));
  } else if (mean_wind_type == "power_law") {
    const double alpha = node["power_law_coefficient"].as<double>();
    windProfile_.reset(new synth_turb::PowerLawProfile(wind_speed, ref_height, alpha));
  } else {
    throw std::runtime_error("SyntheticTurbulence: invalid mean wind type specified = " +
                             mean_wind_type);
  }

  // Smearing factors
  get_required(node, "grid_spacing", gridSpacing_);
  epsilon_ = 2.0 * gridSpacing_;
  get_if_present(node, "gauss_smearing_factor", epsilon_, epsilon_);
  gaussScaling_ = 1.0 / (epsilon_ * std::sqrt(pi));

  // Time offsets if any...
  get_if_present(node, "time_offset", timeOffset_, timeOffset_);

  // Done reading user inputs, process derived data

  // Center of the grid
  turbGrid_.origin_[0] = location[0];
  turbGrid_.origin_[1] = location[1];
  turbGrid_.origin_[2] = location[2];

  // Compute box-fixed reference frame.
  //
  // x-direction points to flow direction (convert from compass direction to vector)
  turbGrid_.trMat_[0][0] = -std::sin(wind_direction);
  turbGrid_.trMat_[0][1] = -std::cos(wind_direction);
  turbGrid_.trMat_[0][2] = 0.0;
  // z always points upwards (for now...)
  turbGrid_.trMat_[2][0] = 0.0;
  turbGrid_.trMat_[2][1] = 0.0;
  turbGrid_.trMat_[2][2] = 1.0;
  // y = z .cross. x
  cross3(&turbGrid_.trMat_[2][0], &turbGrid_.trMat_[0][0], &turbGrid_.trMat_[1][0]);

  NaluEnv::self().naluOutputP0()
    << "Synthethic turbulence forcing initialized \n"
    << "  Turbulence file = " << turbFile_.filename << "\n"
    << "  Box lengths = [" << turbGrid_.boxLen_[0] << ", "
    << turbGrid_.boxLen_[1] << ", " << turbGrid_.boxLen_[2] << "]\n"
    << "  Box dims = [" << turbGrid_.boxDims_[0] << ", "
    << turbGrid_.boxDims_[1] << ", " << turbGrid_.boxDims_[2] << "]\n"
    << "  Grid dx = [" << turbGrid_.dx_[0] << ", " << turbGrid_.dx_[1] << ", "
    << turbGrid_.dx_[2] << "]\n"
    << "  Centroid (forcing plane) = [" << turbGrid_.origin_[0] << ", "
    << turbGrid_.origin_[1] << ", " << turbGrid_.origin_[2] << "]\n"
    << "  Mean wind profile: U = " << windProfile_->reference_velocity()
    << " m/s; Dir = " << wind_direction * 180.0 / pi
    << " deg; H = " << windProfile_->reference_height()
    << " m; type = " << mean_wind_type << std::endl;
}

void SyntheticTurbulence::setup()
{
  auto& meta = realm_.meta_data();

  turbForcingField_ = &(meta.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "synth_turb_forcing"));

  for (auto* part: realm_.interiorPartVec_)
    stk::mesh::put_field_on_mesh(*turbForcingField_, *part, nullptr);
}

void SyntheticTurbulence::execute()
{
  if (isInit_)
    initialize();

  update();
}

void SyntheticTurbulence::initialize()
{
  // Convert current time to an equivalent length based on the reference
  // velocity to determine the position within the turbulence grid
  const double curTime = realm_.get_current_time() - timeOffset_;
  const double eqivLen = windProfile_->reference_velocity() * curTime;
  int il, ir;
  get_lr_indices(turbGrid_, 0, eqivLen, il, ir);
  load_turb_plane_data(turbFile_, turbGrid_, il, ir);

  isInit_ = false;
}

void SyntheticTurbulence::update()
{
  // Convert current time to an equivalent length based on the reference
  // velocity to determine the position within the turbulence grid
  const double curTime = realm_.get_current_time() - timeOffset_;
  const double eqivLen = windProfile_->reference_velocity() * curTime;

  InterpWeights weights;
  get_lr_indices(turbGrid_, 0, eqivLen, weights.il, weights.ir, weights.xl, weights.xr);

  // Check if we need to refresh the planes
  if (weights.il != turbGrid_.iLeft_)
    load_turb_plane_data(turbFile_, turbGrid_, weights.il, weights.ir);

  const auto& meta = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectUnion(realm_.interiorPartVec_);
  const auto bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);

  const auto* coordinates = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());

  double xyzL[SynthTurbTraits::NDimMax]; // coordinates in local frame
  double velL[SynthTurbTraits::NDimMax]; // velocity in local frame
  double velG[SynthTurbTraits::NDimMax]; // velocity in global frame
  for (auto* b: bkts) {
    for (auto node: *b) {
      const double* xyzG = stk::mesh::field_data(*coordinates, node);
      double* forcing = stk::mesh::field_data(*turbForcingField_, node);

      // Transform to local coordinates
      global_to_local(turbGrid_, xyzG, xyzL);

      // Check if the point is in the box, if not we skip this node. The
      // function will also populate the interpolation weights for points that
      // are determined to be within the box.
      bool ptInBox = find_point_in_box(turbGrid_, xyzL, weights);
      if (!ptInBox) continue;

      // Interpolate perturbation velocities in the local reference frame
      interp_perturb_vel(turbGrid_, weights, velL);
      // Transform to global coordinates
      local_to_global_vel(turbGrid_, velL, velG);

      // Based on the equations in http://doi.wiley.com/10.1002/we.1608
      // v_n in Eq. 10
      const double vMag = std::sqrt(velG[0] * velG[0] + velG[1] * velG[1] + velG[2] * velG[2]);
      // (V_n + 1/2 v_n) in Eq. 10
      const double vMagTotal = ((*windProfile_)(xyzG[2]) + 0.5 * vMag);
      // Smearing factor (see Eq. 11). The normal direction to the grid is the
      // x-axis of the local reference frame by construction
      const double term1 = xyzL[0] / epsilon_;
      const double eta = std::exp(-(term1 * term1)) * gaussScaling_;
      const double factor = vMagTotal * eta / gridSpacing_;

      // Defer density and volume scaling to the nodal assembly algorithm
      forcing[0] = velG[0] * factor;
      forcing[1] = velG[1] * factor;
      forcing[2] = velG[2] * factor;
    }
  }
}

}  // nalu
}  // sierra
