/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SYNTHETICTURBULENCE_H
#define SYNTHETICTURBULENCE_H

#include <string>
#include <cmath>
#include <memory>

#include "KokkosInterface.h"
#include "FieldTypeDef.h"
#include "yaml-cpp/yaml.h"

namespace sierra {
namespace nalu {

namespace synth_turb {
class MeanProfile
{
public:
  MeanProfile(double refVel, double refHeight)
    : refVel_(refVel),
      refHeight_(refHeight)
  {}

  virtual ~MeanProfile() = default;

  virtual double operator()(double /* height */) const
  { return refVel_; }

  inline double reference_velocity() const { return refVel_; }

  inline double reference_height() const { return refHeight_; }

protected:
  const double refVel_;
  const double refHeight_;
};

}

class Realm;

struct SynthTurbTraits
{
  static constexpr int NDimMax{3};
  using ArrayType = double[NDimMax];
  using IntArrayType = int[NDimMax];
  using TransMatType = double[NDimMax][NDimMax];
  // using StructField = Kokkos::View<double*, Kokkos::LayoutRight, MemSpace>;
  using StructField = std::vector<double>;
};

struct SynthTurbData
{
  // Dimensions of the box
  SynthTurbTraits::IntArrayType boxDims_;

  // Length of the boxes in each direction
  SynthTurbTraits::ArrayType boxLen_;

  SynthTurbTraits::ArrayType dx_;

  // Reference point for the turbulence box. Reference is the mid-point of the
  // turbulence grid at the plane where it is injected into the CFD flow field.
  SynthTurbTraits::ArrayType origin_;

  // Transformation matrix to convert from global coordinate system to local
  // coordinate system
  SynthTurbTraits::TransMatType trMat_;

  // Perturbation velocities (2, ny, nz)
  SynthTurbTraits::StructField uvel_;
  SynthTurbTraits::StructField vvel_;
  SynthTurbTraits::StructField wvel_;
  // SynthTurbTraits::StructField::HostMirror h_uvel_;
  // SynthTurbTraits::StructField::HostMirror h_vvel_;
  // SynthTurbTraits::StructField::HostMirror h_wvel_;

  // Indices of the two planes stored in the data arrays
  int iLeft_;
  int iRight_;
};

class SyntheticTurbulence
{
public:
  struct NCBoxTurb
  {
    std::string filename;

    // NetCDF file ID
    int ncid;

    // Dimensions
    int sDim, xDim, yDim, zDim;

    // Perturbation velocity field IDs
    int uid, vid, wid;

    // Box length and grid size IDs
    int boxLenid, dxid;

    // Scale and divergence correction IDs
    int scaleid, divCorrid;
  };

  SyntheticTurbulence(Realm&);

  SyntheticTurbulence() = delete;
  SyntheticTurbulence(const SyntheticTurbulence&) = delete;
  SyntheticTurbulence& operator=(const SyntheticTurbulence&) = delete;

  ~SyntheticTurbulence() = default;

  void load(const YAML::Node&);

  void setup();

  void execute();

private:
  void initialize();

  void update();

  Realm& realm_;

  // Turbulence file read information
  NCBoxTurb turbFile_;

  // Turbulence box data
  SynthTurbData turbGrid_;

  VectorFieldType* turbForcingField_;

  std::unique_ptr<synth_turb::MeanProfile> windProfile_;

  double gridSpacing_;
  double epsilon_;
  double gaussScaling_;

  double timeOffset_{0.0};

  bool isInit_{true};
};

}  // nalu
}  // sierra


#endif /* SYNTHETICTURBULENCE_H */
