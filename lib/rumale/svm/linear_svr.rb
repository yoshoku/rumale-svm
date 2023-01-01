# frozen_string_literal: true

require 'numo/libsvm'
require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/validation'

module Rumale
  module SVM
    # LinearSVR is a class that provides Support Vector Regressor in LIBLINEAR with Rumale interface.
    #
    # @example
    #   estimator = Rumale::SVM::LinearSVR.new(reg_param: 1.0, random_seed: 1)
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    class LinearSVR < Rumale::Base::Estimator
      include Rumale::Base::Regressor

      # Return the weight vector for LinearSVR.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for LinearSVR.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :bias_term

      # Create a new regressor with Support Vector Regressor.
      #
      # @param loss [String] The type of loss function ('squared_epsilon_insensitive' or 'epsilon_insensitive').
      # @param dual [Boolean] The flag indicating whether to solve dual optimization problem.
      #   When n_samples > n_features, dual = false is more preferable.
      #   This parameter is ignored if loss = 'epsilon_insensitive'.
      # @param reg_param [Float] The regularization parameter.
      # @param epsilon [Float] The epsilon parameter in loss function of espsilon-svr.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      #   This parameter is ignored if fit_bias = false.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output learning process message
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator.
      def initialize(loss: 'squared_epsilon_insensitive', dual: true, reg_param: 1.0, epsilon: 0.1,
                     fit_bias: true, bias_scale: 1.0, tol: 1e-3, verbose: false, random_seed: nil)
        super()
        @params = {}
        @params[:loss] = loss == 'epsilon_insensitive' ? 'epsilon_insensitive' : 'squared_epsilon_insensitive'
        @params[:dual] = dual
        @params[:reg_param] = reg_param.to_f
        @params[:epsilon] = epsilon.to_f
        @params[:fit_bias] = fit_bias
        @params[:bias_scale] = bias_scale.to_f
        @params[:tol] = tol.to_f
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed.nil? ? nil : random_seed.to_i
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [LinearSVR] The learned regressor itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_target_value_array(y)
        Rumale::Validation.check_sample_size(x, y)
        xx = fit_bias? ? expand_feature(x) : x
        @model = Numo::Liblinear.train(xx, y, liblinear_params)
        @weight_vec, @bias_term = weight_and_bias(@model[:w])
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::DFloat] (shape: [n_samples]) Predicted value per sample.
      def predict(x)
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
        xx = fit_bias? ? expand_feature(x) : x
        Numo::Liblinear.predict(xx, liblinear_params, @model)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LinearSVR.
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

      def expand_feature(x)
        n_samples = x.shape[0]
        Numo::NArray.hstack([x, Numo::DFloat.ones([n_samples, 1]) * bias_scale])
      end

      def weight_and_bias(base_weight)
        bias_vec = 0.0
        weight_mat = base_weight.dup
        if fit_bias?
          bias_vec = weight_mat[-1]
          weight_mat = weight_mat[0...-1].dup
        end
        [weight_mat, bias_vec]
      end

      def liblinear_params
        res = {}
        res[:solver_type] = solver_type
        res[:eps] = @params[:tol]
        res[:C] = @params[:reg_param]
        res[:p] = @params[:epsilon]
        res[:verbose] = @params[:verbose]
        res[:random_seed] = @params[:random_seed]
        res
      end

      def solver_type
        return Numo::Liblinear::SolverType::L2R_L1LOSS_SVR_DUAL if @params[:loss] == 'epsilon_insensitive'
        return Numo::Liblinear::SolverType::L2R_L2LOSS_SVR_DUAL if @params[:dual]

        Numo::Liblinear::SolverType::L2R_L2LOSS_SVR
      end

      def fit_bias?
        @params[:fit_bias]
      end

      def bias_scale
        @params[:bias_scale]
      end

      def trained?
        !@model.nil?
      end
    end
  end
end
