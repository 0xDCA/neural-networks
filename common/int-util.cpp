#include "int-util.h"
#include <stdexcept>

using namespace std;

int int_division_round_up(int a, int b) {
	if (b == 0) {
		throw runtime_error("Division by 0");
	}

	if ((a >= 0) != (b >= 0)) {
		return a / b;
	}

	if (a < 0) {
		a = -a;
		b = -b;
	}

	return (a + b - 1) / b;
}

