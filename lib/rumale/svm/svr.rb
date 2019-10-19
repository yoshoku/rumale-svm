# frozen_string_literal: true

require 'numo/libsvm'
require 'rumale/base/base_estimator'
require 'rumale/base/regressor'

module Rumale
  module SVM
    # SVR is a class that provides Rumale interface for Kernel Epsilon-Support Vector Regressor in LIBSVM.
    #
    # @example
    #   estimator = Rumale::SVM::SVR.new(reg_param: 1.0, kernel: 'rbf', gamma: 10.0, random_seed: 1)
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    class SVR
      include Base::BaseEstimator
      include Base::Regressor

      # Create a new classifier with Kernel Epsilon-Support Vector Regressor.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param epsilon [Float] The epsilon parameter in loss function of espsilon-svr.
      # @param kernel [String] The type of kernel function ('rbf', 'linear', 'poly', 'sigmoid', and 'precomputed').
      # @param degree [Integer] The degree parameter in polynomial kernel function.
      # @param gamma [Float] The gamma parameter in rbf/poly/sigmoid kernel function.
      # @param coef0 [Float] The coefficient in poly/sigmoid kernel function.
      # @param shrinking [Boolean] The flag indicating whether to use the shrinking heuristics.
      # @param cache_size [Float] The cache memory size in MB.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output learning process message
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, epsilon: 0.1, kernel: 'rbf', degree: 3, gamma: 1.0, coef0: 0.0,
                     shrinking: true, cache_size: 200.0, tol: 1e-3, verbose: false, random_seed: nil)
        check_params_float(reg_param: reg_param, epsilon: epsilon, gamma: gamma, coef0: coef0, cache_size: cache_size, tol: tol)
        check_params_integer(degree: degree)
        check_params_boolean(shrinking: shrinking, verbose: verbose)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:epsilon] = epsilon
        @params[:kernel] = kernel
        @params[:degree] = degree
        @params[:gamma] = gamma
        @params[:coef0] = coef0
        @params[:shrinking] = shrinking
        @params[:cache_size] = cache_size
        @params[:tol] = tol
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed
        @model = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   If the kernel is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [SVR] The learned regressor itself.
      def fit(x, y)
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        xx = precomputed_kernel? ? add_index_col(x) : x
        @model = Numo::Libsvm.train(xx, y, libsvm_params)
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      #   If the kernel is 'precomputed', the shape of x must be [n_samples, n_training_samples].
      # @return [Numo::DFloat] (shape: [n_samples]) Predicted value per sample.
      def predict(x)
        check_sample_array(x)
        xx = precomputed_kernel? ? add_index_col(x) : x
        Numo::Libsvm.predict(xx, libsvm_params, @model)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about SVR.
      def marshal_dump
        { params: @params,
          model: @model }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @model = obj[:model]
        nil
      end

      # Return the indices of support vectors.
      # @return [Numo::Int32] (shape: [n_support_vectors])
      def support
        @model[:sv_indices]
      end

      # Return the support_vectors.
      # @return [Numo::DFloat] (shape: [n_support_vectors, n_features])
      def support_vectors
        precomputed_kernel? ? del_index_col(@model[:SV]) : @model[:SV]
      end

      # Return the number of support vectors.
      # @return [Integer]
      def n_support
        support.size
      end

      # Return the coefficients of the support vector in decision function.
      # @return [Numo::DFloat] (shape: [1, n_support_vectors])
      def duel_coef
        @model[:sv_coef]
      end

      # Return the intercepts in decision function.
      # @return [Numo::DFloat] (shape: [1])
      def intercept
        @model[:rho]
      end

      private

      def add_index_col(x)
        idx = Numo::Int32.new(x.shape[0]).seq + 1
        Numo::NArray.hstack([idx.expand_dims(1), x])
      end

      def del_index_col(x)
        x[true, 1..-1].dup
      end

      def precomputed_kernel?
        @params[:kernel] == 'precomputed'
      end

      def libsvm_params
        res = @params.merge(svm_type: Numo::Libsvm::SvmType::EPSILON_SVR)
        res[:kernel_type] = case res.delete(:kernel)
                            when 'linear'
                              Numo::Libsvm::KernelType::LINEAR
                            when 'poly'
                              Numo::Libsvm::KernelType::POLY
                            when 'sigmoid'
                              Numo::Libsvm::KernelType::SIGMOID
                            when 'precomputed'
                              Numo::Libsvm::KernelType::PRECOMPUTED
                            else
                              Numo::Libsvm::KernelType::RBF
                            end
        res[:C] = res.delete(:reg_param)
        res[:p] = res.delete(:epsilon)
        res[:eps] = res.delete(:tol)
        res
      end
    end
  end
end
