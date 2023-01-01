# frozen_string_literal: true

require 'numo/libsvm'
require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/validation'

module Rumale
  module SVM
    # NuSVR is a class that provides Kernel Nu-Support Vector Regressor in LIBSVM with Rumale interface.
    #
    # @example
    #   estimator = Rumale::SVM::NuSVR.new(nu: 0.5, kernel: 'rbf', gamma: 10.0, random_seed: 1)
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    class NuSVR < Rumale::Base::Estimator
      include Rumale::Base::Regressor

      # Create a new regressor with Kernel Nu-Support Vector Regressor.
      #
      # @param nu [Float] The regularization parameter. The interval of nu is (0, 1].
      # @param kernel [String] The type of kernel function ('rbf', 'linear', 'poly', 'sigmoid', and 'precomputed').
      # @param degree [Integer] The degree parameter in polynomial kernel function.
      # @param gamma [Float] The gamma parameter in rbf/poly/sigmoid kernel function.
      # @param coef0 [Float] The coefficient in poly/sigmoid kernel function.
      # @param shrinking [Boolean] The flag indicating whether to use the shrinking heuristics.
      # @param cache_size [Float] The cache memory size in MB.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output learning process message
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator.
      def initialize(nu: 0.5, kernel: 'rbf', degree: 3, gamma: 1.0, coef0: 0.0,
                     shrinking: true, cache_size: 200.0, tol: 1e-3, verbose: false, random_seed: nil)
        super()
        @params = {}
        @params[:nu] = nu.to_f
        @params[:kernel] = kernel
        @params[:degree] = degree.to_i
        @params[:gamma] = gamma.to_f
        @params[:coef0] = coef0.to_f
        @params[:shrinking] = shrinking
        @params[:cache_size] = cache_size.to_f
        @params[:tol] = tol.to_f
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed.nil? ? nil : random_seed.to_i
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   If the kernel is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [NuSVR] The learned regressor itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_target_value_array(y)
        Rumale::Validation.check_sample_size(x, y)
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
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
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
        res[:eps] = res.delete(:tol)
        res
      end

      def trained?
        !@model.nil?
      end
    end
  end
end
