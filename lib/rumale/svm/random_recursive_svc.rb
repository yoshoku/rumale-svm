# frozen_string_literal: true

require 'rumale/utils'
require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/svm/linear_svc'

module Rumale
  module SVM
    # RandomRecursiveSVC is a class that implements Random Recursive Support Vector Classifier.
    #
    # @example
    #   require 'rumale/svm'
    #
    #   estimator = Rumale::SVM::RandomRecursiveSVC.new(n_hidden_layers: 2, beta: 0.5, random_seed: 1)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Vinyals, O., Jia, Y., Deng, L., and Darrell, T., "Learning with Recursive Perceptual Representations," In Proc. NIPS'12, pp. 2825--2833, 2012.
    class RandomRecursiveSVC < Rumale::Base::Estimator
      include Rumale::Base::Classifier

      # Return the classifiers for each layer.
      # @return [Array<LinearSVC>]
      attr_reader :classifiers

      # Return the random matrices for each hidden layer.
      # @return [Array<Numo::DFloat>] (shape: [n_classes, n_features])
      attr_reader :random_matrices

      # Create a new classifier with Random Recursive Support Vector Machine.
      #
      # @param n_hidden_layers [Integer] The number of hidden layers.
      # @param beta [Float] The weight parameter for the degree of moving the original data.
      # @param penalty [String] The type of norm used in the penalization ('l2' or 'l1').
      # @param loss [String] The type of loss function ('squared_hinge' or 'hinge').
      #   This parameter is ignored if penalty = 'l1'.
      # @param dual [Boolean] The flag indicating whether to solve dual optimization problem.
      #   When n_samples > n_features, dual = false is more preferable.
      #   This parameter is ignored if loss = 'hinge'.
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      #   This parameter is ignored if fit_bias = false.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output learning process message
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator.
      def initialize(n_hidden_layers: 2, beta: 0.5, penalty: 'l2', loss: 'squared_hinge', dual: true, reg_param: 1.0,
                     fit_bias: true, bias_scale: 1.0, tol: 1e-3, verbose: false, random_seed: nil)
        super()
        @params = {
          n_hidden_layers: n_hidden_layers,
          beta: beta,
          penalty: penalty == 'l1' ? 'l1' : 'l2',
          loss: loss == 'hinge' ? 'hinge' : 'squared_hinge',
          dual: dual,
          reg_param: reg_param.to_f,
          fit_bias: fit_bias,
          bias_scale: bias_scale.to_f,
          tol: tol.to_f,
          verbose: verbose,
          random_seed: random_seed || Random.rand(4_294_967_295)
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [RandomRecursiveSVC] The learned classifier itself.
      def fit(x, y)
        partial_fit(x, y)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        @classifiers.last.decision_function(transform(x))
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        @classifiers.last.predict(transform(x))
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_features]) The output of last hidden layer.
      def transform(x)
        d = x
        s = Numo::DFloat.zeros(x.shape)
        @random_matrices.each_with_index do |w, n|
          o = @classifiers[n].predict_proba(d)
          s += o.dot(w)
          d = sigmoid(x + @params[:beta] * s)
        end
        d
      end

      private

      def partial_fit(x, y)
        @classifiers = []
        @random_matrices = []
        sub_rng = @rng.dup
        n_classes = y.to_a.uniq.size
        n_features = x.shape[1]
        d = x
        s = Numo::DFloat.zeros(x.shape)
        @params[:n_hidden_layers].times do
          svc = LinearSVC.new(**linear_svc_params).fit(d, y)
          o = svc.predict_proba(d)
          w = ::Rumale::Utils.rand_normal([n_classes, n_features], sub_rng)
          s += o.dot(w)
          d = sigmoid(x + @params[:beta] * s)
          @classifiers << svc
          @random_matrices << w
        end
        svc = LinearSVC.new(**linear_svc_params).fit(d, y)
        @classifiers << svc
      end

      def linear_svc_params
        @params.reject { |key, _| RRSVC_PARAMS.include?(key) }.merge(probability: true)
      end

      def sigmoid(x)
        0.5 * (Numo::NMath.tanh(0.5 * x) + 1.0)
      end

      RRSVC_PARAMS = %i[n_hidden_layers beta].freeze

      private_constant :RRSVC_PARAMS
    end
  end
end
