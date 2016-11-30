
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <endian.h>

using namespace std;

double sigmoid(double t) {
    return 1.0 / (1 + exp(-t));
}
