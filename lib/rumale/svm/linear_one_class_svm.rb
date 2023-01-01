# frozen_string_literal: true

require 'numo/liblinear'
require 'rumale/base/estimator'
require 'rumale/validation'

module Rumale
  module SVM
    # LinearOneClassSVM is a class that provides linear One-class Support Vector Machine in LIBLINEAR with Rumale interface.
    #
    # @example
    #   estimator = Rumale::SVM::LinearOneClassSVM.new(nu: 0.05, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    class LinearOneClassSVM < Rumale::Base::Estimator
      # Return the weight vector for LinearOneClassSVM.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for LinearOneClassSVM.
      # @return [Float]
      attr_reader :bias_term

      # Create a new estimator with linear One-class Support Vector Machine.
      #
      # @param nu [Float] The fraction of data as outliers. The interval of nu is (0, 1].
      # @param reg_param [Float] The regularization parameter.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output learning process message
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator.
      def initialize(nu: 0.05, reg_param: 1.0, tol: 1e-3, verbose: false, random_seed: nil)
        super()
        @params = {}
        @params[:nu] = nu.to_f
        @params[:reg_param] = reg_param.to_f
        @params[:tol] = tol.to_f
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed.nil? ? nil : random_seed.to_i
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> LinearOneClassSVM
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #
      # @return [LinearOneClassSVM] The learned estimator itself.
      def fit(x, _y = nil)
        x = Rumale::Validation.check_convert_sample_array(x)
        dummy = Numo::DFloat.ones(x.shape[0])
        @model = Numo::Liblinear.train(x, dummy, liblinear_params)
        @weight_vec = @model[:w].dup
        @bias_term = @model[:rho]
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
        Numo::Liblinear.decision_function(x, liblinear_params, @model)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted label per sample.
      def predict(x)
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
        Numo::Int32.cast(Numo::Liblinear.predict(x, liblinear_params, @model))
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LinearOneClassSVM.
      def marshal_dump
        { params: @params,
          model: @model,
          weight_vec: @weight_vec,
          bias_term: @bias_term }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @model = obj[:model]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        nil
      end

      private

      def liblinear_params
        res = {}
        res[:solver_type] = Numo::Liblinear::SolverType::ONECLASS_SVM
        res[:eps] = @params[:tol]
        res[:C] = @params[:reg_param]
        res[:nu] = @params[:nu]
        res[:verbose] = @params[:verbose]
        res[:random_seed] = @params[:random_seed]
        res
      end

      def trained?
        !@model.nil?
      end
    end
  end
end
