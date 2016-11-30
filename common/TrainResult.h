/*
 * TrainResult.h
 *
 *  Created on: 20/10/2016
 *      Author: drcastroa
 */

#ifndef TRAINRESULT_H_
#define TRAINRESULT_H_

struct TrainResult {
    TrainResult(int iterations, double error) : iterations(iterations), error(error) {}

    int iterations;
    double error;
};

#endif /* TRAINRESULT_H_ */
