library(tidyverse)
library(jsonlite)

prepData <- function(data) {
    return(
        data %>%
            mutate(
                timeS = (time - time[1])/(1000 * 1000), yawRateRotPerS = gyroADC.2./360,
                pitchRateRotPerS = gyroADC.1./360, rollRateRotPerS = gyroADC.0./360,
                frontLeftRPS = eRPM.0./6, backLeftRPS = eRPM.1./6, frontRightRPS = eRPM.2./6,
                backRightRPS = eRPM.3./6, frontLeftVolts = motor.0./(2047 * 100) * vbatLatest,
                backLeftVolts = motor.1./(2047 * 100) * vbatLatest, frontRightVolts = motor.2./(2047 * 100) *
                  vbatLatest, backRightVolts = motor.3./(2047 * 100) * vbatLatest,
            ) %>%
            filter(
                timeS < max(timeS) -
                  5
            )
    )
}

signedSquare <- function(x) {
    return(
        sign(x) *
            x^2
    )
}

diff <- function(value, n) {
    return(
        value[-(1:n)] - value[1:(length(value) -
            n)]
    )
}

centralFiniteDiff <- function(value, time, n) {
    return(diff(value, n)/diff(time, n))
}

symmetricTruncate <- function(value, n) {
    return(
        value[(n/2):(length(value) -
            n/2 - 1)]
    )
}

propForceMean <- function(frontLeft, frontRight, backLeft, backRight, weights) {
    return(weights[0] * frontLeft)
}

fitYaw <- function(data, window, threshold) {
    yawData <- data.frame(
        time = data$timeS %>%
            symmetricTruncate(window),
        yawRate = data$yawRateRotPerS %>%
            symmetricTruncate(window),
        yawAccel = data$yawRateRotPerS %>%
            centralFiniteDiff(data$timeS, window),
        propVelSum = (-data$frontLeftRPS + data$frontRightRPS + data$backLeftRPS -
            data$backRightRPS) %>%
            symmetricTruncate(window),
        propVelSumSq = (-signedSquare(data$frontLeftRPS) +
            signedSquare(data$frontRightRPS) +
            signedSquare(data$backLeftRPS) -
            signedSquare(data$backRightRPS)) %>%
            symmetricTruncate(window),
        propAccelSum = -data$frontLeftRPS %>%
            centralFiniteDiff(data$timeS, window) +
            data$frontRightRPS %>%
                centralFiniteDiff(data$timeS, window) +
            data$backLeftRPS %>%
                centralFiniteDiff(data$timeS, window) -
            data$backRightRPS %>%
                centralFiniteDiff(data$timeS, window)
    ) %>%
        filter(
            abs(yawRate) >
                max(abs(yawRate)) *
                  threshold
        )

    return(lm(yawAccel ~ 0 + yawRate + propVelSumSq + propAccelSum, yawData))
}

fitPitch <- function(data, window, threshold, dFront, dBack) {

    pitchData <- data.frame(
        time = data$timeS %>%
            symmetricTruncate(window),
        pitchRate = data$pitchRateRotPerS %>%
            symmetricTruncate(window),
        pitchAccel = data$pitchRateRotPerS %>%
            centralFiniteDiff(data$timeS, window),
        propVelSum = ((2 * dFront * (data$frontLeftRPS + data$frontRightRPS) - 2 * dBack *
            (data$backLeftRPS + data$backRightRPS))/(dFront + dBack)) %>%
            symmetricTruncate(window),
        propVelSumSq = ((2 * dFront * (signedSquare(data$frontLeftRPS) +
            signedSquare(data$frontRightRPS)) -
            2 * dBack * (signedSquare(data$backLeftRPS) +
                signedSquare(data$backRightRPS)))/(dFront +
            dBack)) %>%
            symmetricTruncate(window)
    ) %>%
        filter(
            abs(pitchRate) >
                max(abs(pitchRate)) *
                  threshold
        )

    return(lm(pitchAccel ~ 0 + pitchRate + propVelSumSq, pitchData))
}

fitRoll <- function(data, window, threshold, dLeft, dRight) {
    rollData <- data.frame(
        time = data$timeS %>%
            symmetricTruncate(window),
        rollRate = data$rollRateRotPerS %>%
            symmetricTruncate(window),
        rollAccel = data$rollRateRotPerS %>%
            centralFiniteDiff(data$timeS, window),
        propVelSum = ((2 * dRight * (data$frontRightRPS + data$backRightRPS) - 2 * dLeft *
            (data$frontLeftRPS + data$backLeftRPS))/(dLeft + dRight)) %>%
            symmetricTruncate(window),
        propVelSumSq = ((2 * dRight * (signedSquare(data$frontRightRPS) +
            signedSquare(data$backRightRPS)) -
            2 * dLeft * (signedSquare(data$frontLeftRPS) +
                signedSquare(data$backLeftRPS)))/(dLeft +
            dRight)) %>%
            symmetricTruncate(window)
    ) %>%
        filter(
            abs(rollRate) >
                max(abs(rollRate)) *
                  threshold
        )

    return(lm(rollAccel ~ 0 + rollRate + propVelSumSq, rollData))
}

fitMotors <- function(data, window, threshold) {
    motorData <- data.frame(
        time = rep(
            data$timeS %>%
                symmetricTruncate(window),
            4
        ),
        volts = c(
            data$frontLeftVolts %>%
                symmetricTruncate(window),
            data$frontRightVolts %>%
                symmetricTruncate(window),
            data$backLeftVolts %>%
                symmetricTruncate(window),
            data$backRightVolts %>%
                symmetricTruncate(window)
        ),
        motorVel = c(
            data$frontLeftRPS %>%
                symmetricTruncate(window),
            data$frontRightRPS %>%
                symmetricTruncate(window),
            data$backLeftRPS %>%
                symmetricTruncate(window),
            data$backRightRPS %>%
                symmetricTruncate(window)
        ),
        motorAccel = c(
            data$frontLeftRPS %>%
                centralFiniteDiff(data$timeS, window),
            data$frontRightRPS %>%
                centralFiniteDiff(data$timeS, window),
            data$backLeftRPS %>%
                centralFiniteDiff(data$timeS, window),
            data$backRightRPS %>%
                centralFiniteDiff(data$timeS, window)
        )
    ) %>%
        mutate(motorVelSq = signedSquare(motorVel)) %>%
        filter(
            (abs(motorVel) >
                max(abs(motorVel)) *
                  threshold)
        )

    return(lm(volts ~ 0 + motorVel + motorVelSq + motorAccel, motorData))
}

data <- prepData(read.csv("bf-4.6-BTFL_BLACKBOX_LOG_AOS4_20250731_084145_SKYSTARSF7HDPRO.BBL.csv"))

# Window size for first-order finite difference
yawWindow <- 50
# Motion threshold for discarding data without sufficient excitation for fit
yawThreshold <- 0.3
pitchWindow <- 20
pitchThreshold <- 0.3
rollWindow <- 20
rollThreshold <- 0.3
motorWindow <- 16
motorThreshold <- 0.1
dFront <- 1
dBack <- 1.7
dLeft <- 1
dRight <- 1

yawFit <- fitYaw(data, yawWindow, yawThreshold)
pitchFit <- fitPitch(data, pitchWindow, pitchThreshold, dFront, dBack)
rollFit <- fitRoll(data, rollWindow, rollThreshold, dLeft, dRight)
motorFit <- fitMotors(data, motorWindow, motorThreshold)

summary(yawFit)
summary(pitchFit)
summary(rollFit)
summary(motorFit)

fits <- list(
    yaw = as.list(yawFit$coefficients),
    pitch = as.list(pitchFit$coefficients),
    roll = as.list(rollFit$coefficients),
    motors = as.list(motorFit$coefficients)
)

write_json(fits, 'regression_coefficients.json', digits = 8, pretty = TRUE, auto_unbox = TRUE)