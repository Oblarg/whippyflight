#pragma once

#include "units.hpp"
#include "types.hpp"
#include "gain_generation.hpp"
#include <array>

namespace lqr {

// Legacy function for backward compatibility
InterpolatedModel lerp_lqr(const MotorSpeeds& motor_speeds, const GainLookup& lut);

} // namespace lqr 