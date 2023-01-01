# frozen_string_literal: true

require 'numo/liblinear'
require 'rumale/base/estimator'
require 'rumale/base/classifier'

module Rumale
  module SVM
    # LogisticRegression is a class that provides Logistic Regression in LIBLINEAR with Rumale interface
    #
    # @example
    #   estimator = Rumale::SVM::LogisticRegression.new(penalty: 'l2', dual: false, reg_param: 1.0, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    class LogisticRegression < Rumale::Base::Estimator
      include Rumale::Base::Classifier

      # Return the weight vector for LogisticRegression.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for LogisticRegression.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :bias_term

      # Create a new classifier with Logistic Regression.
      #
      # @param penalty [String] The type of norm used in the penalization ('l2' or 'l1').
      # @param dual [Boolean] The flag indicating whether to solve dual optimization problem.
      #   When n_samples > n_features, dual = false is more preferable.
      #   This parameter is ignored if penalty = 'l1'.
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      #   This parameter is ignored if fit_bias = false.
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output learning process message
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator.
      def initialize(penalty: 'l2', dual: true, reg_param: 1.0,
                     fit_bias: true, bias_scale: 1.0, tol: 1e-3, verbose: false, random_seed: nil)
        super()
        @params = {}
        @params[:penalty] = penalty == 'l1' ? 'l1' : 'l2'
        @params[:dual] = dual
        @params[:reg_param] = reg_param.to_f
        @params[:fit_bias] = fit_bias
        @params[:bias_scale] = bias_scale.to_f
        @params[:tol] = tol.to_f
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed.nil? ? nil : random_seed.to_i
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [LogisticRegression] The learned classifier itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_label_array(y)
        Rumale::Validation.check_sample_size(x, y)
        xx = fit_bias? ? expand_feature(x) : x
        @model = Numo::Liblinear.train(xx, y, liblinear_params)
        @weight_vec, @bias_term = weight_and_bias(@model[:w])
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
        xx = fit_bias? ? expand_feature(x) : x
        Numo::Liblinear.decision_function(xx, liblinear_params, @model)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
        xx = fit_bias? ? expand_feature(x) : x
        Numo::Int32.cast(Numo::Liblinear.predict(xx, liblinear_params, @model))
      end

      # Predict class probability for samples.
      # This method works correctly only if the probability parameter is true.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        raise "#{self.class.name}##{__method__} expects to be called after training the model with the fit method." unless trained?
        x = Rumale::Validation.check_convert_sample_array(x)
        xx = fit_bias? ? expand_feature(x) : x
        Numo::Liblinear.predict_proba(xx, liblinear_params, @model)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LogisticRegression.
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
        if binary_class?
          bias_vec = 0.0
          weight_mat = base_weight.dup
          if fit_bias?
            bias_vec = weight_mat[-1]
            weight_mat = weight_mat[0...-1].dup
          end
        else
          bias_vec = Numo::DFloat.zeros(n_classes)
          weight_mat = base_weight.reshape(n_features, n_classes).transpose.dup
          if fit_bias?
            bias_vec = weight_mat[true, -1].dup
            weight_mat = weight_mat[true, 0...-1].dup
          end
        end
        [weight_mat, bias_vec]
      end

      def liblinear_params
        res = {}
        res[:solver_type] = solver_type
        res[:eps] = @params[:tol]
        res[:C] = @params[:reg_param]
        res[:verbose] = @params[:verbose]
        res[:random_seed] = @params[:random_seed]
        res
      end

      def solver_type
        return Numo::Liblinear::SolverType::L1R_LR if @params[:penalty] == 'l1'
        return Numo::Liblinear::SolverType::L2R_LR_DUAL if @params[:dual]

        Numo::Liblinear::SolverType::L2R_LR
      end

      def binary_class?
        @model[:nr_class] == 2
      end

      def fit_bias?
        @params[:fit_bias]
      end

      def bias_scale
        @params[:bias_scale]
      end

      def n_classes
        @model[:nr_class]
      end

      def n_features
        @model[:nr_feature]
      end

      def trained?
        !@model.nil?
      end
    end
  end
end
