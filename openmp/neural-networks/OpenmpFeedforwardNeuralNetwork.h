#ifndef OPENMPFEEDFORWARDNEURALNETWORK_H
#define OPENMPFEEDFORWARDNEURALNETWORK_H

#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <exception>

#include "TrainResult.h"
#include "ManagedMatrix.h"
#include "FeedforwardNeuralNetwork.h"

struct TrainSettings;
struct WorkerParams {
    const Eigen::MatrixXd* x;
    const Eigen::MatrixXd* y;
    const TrainSettings* train_settings;
    std::vector<Eigen::MatrixXd>* weights;
    unsigned int seed;
};

class OpenmpFeedforwardNeuralNetwork : public FeedforwardNeuralNetwork {
public:
    OpenmpFeedforwardNeuralNetwork(const std::vector<int>& layers);

    Eigen::VectorXd predict(const Eigen::VectorXd& input) const;

    void set_weights_eigen(int source_layer, const Eigen::MatrixXd& weights);
    Eigen::MatrixXd get_weights_eigen(int source_layer) const;

    TrainResult train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y, const TrainSettings& train_settings);

    double compute_error(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double regularization_term) const;

    std::vector<Eigen::MatrixXd> compute_weights_error(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const;

    virtual std::vector<std::vector<double> > get_activations(const std::vector<double>& input) const;

    virtual std::vector<double> predict(const std::vector<double>& input) const;

    virtual std::vector<dc::ManagedMatrix<double> > compute_weights_error(const std::vector<double>& input,
        const std::vector<double>& output) const;

    virtual TrainResult train(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
        const TrainSettings& train_settings);

    virtual void set_weights(int source_layer, const dc::ManagedMatrix<double> &weights);

    virtual dc::ManagedMatrix<double> get_weights(int source_layer) const;

    virtual double compute_error(const dc::ManagedMatrix<double>& x, const dc::ManagedMatrix<double>& y,
        double regularization_term);
private:
    std::vector<int> layers;
    std::vector<Eigen::MatrixXd> weight_list;

    std::vector<Eigen::VectorXd> forward_propagation(const Eigen::VectorXd& input) const;

    Eigen::MatrixXd random_matrix(int rows, int cols, double epsilon, std::mt19937& generator);
    void back_propagation(const std::vector<Eigen::VectorXd>& fp_results, const Eigen::VectorXd& y,
                          std::vector<Eigen::MatrixXd>& out) const;

    static std::vector<Eigen::VectorXd> forward_propagation(const Eigen::VectorXd& input,
                                                            const std::vector<Eigen::MatrixXd>& weight_list);
    static void back_propagation(const std::vector<Eigen::VectorXd>& fp_results, const Eigen::VectorXd& y,
                          const std::vector<Eigen::MatrixXd>& weight_list,
                          std::vector<Eigen::MatrixXd>& out);

    static void do_gradient_descent(const WorkerParams& params);
    static double compute_error(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double regularization_term,
      const std::vector<Eigen::MatrixXd>& weight_list);
};


#endif //OPENMPFEEDFORWARDNEURALNETWORK_H
