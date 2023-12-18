# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/pairwise_metric'
require 'rumale/validation'
require 'rumale/svm/linear_svc'

module Rumale
  module SVM
    # ClusteredSVC is a class that implements Clustered Support Vector Classifier.
    #
    # @example
    #   require 'rumale/svm'
    #
    #   estimator = Rumale::SVM::ClusteredSVC.new(n_clusters: 16, reg_param_global: 1.0, random_seed: 1)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Gu, Q., and Han, J., "Clustered Support Vector Machines," In Proc. AISTATS'13, pp. 307--315, 2013.
    class ClusteredSVC < Rumale::Base::Estimator
      include Rumale::Base::Classifier

      # Return the classifier.
      # @return [LinearSVC]
      attr_reader :model

      # Return the centroids.
      # @return [Numo::DFloat] (shape: [n_clusters, n_features])
      attr_accessor :cluster_centers

      # Create a new classifier with Random Recursive Support Vector Machine.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param reg_param_global [Float] The regularization parameter for global reference vector.
      # @param max_iter_kmeans [Integer] The maximum number of iterations for k-means clustering.
      # @param tol_kmeans [Float] The tolerance of termination criterion for k-means clustering.
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
      def initialize(n_clusters: 8, reg_param_global: 1.0, max_iter_kmeans: 100, tol_kmeans: 1e-6, # rubocop:disable Metrics/ParameterLists
                     penalty: 'l2', loss: 'squared_hinge', dual: true, reg_param: 1.0,
                     fit_bias: true, bias_scale: 1.0, tol: 1e-3, verbose: false, random_seed: nil)
        super()
        @params = {
          n_clusters: n_clusters,
          reg_param_global: reg_param_global,
          max_iter_kmeans: max_iter_kmeans,
          tol_kmeans: tol_kmeans,
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
        @cluster_centers = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [ClusteredSVC] The learned classifier itself.
      def fit(x, y)
        z = transform(x)
        @model = LinearSVC.new(**linear_svc_params).fit(z, y)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        z = transform(x)
        @model.decision_function(z)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        z = transform(x)
        @model.predict(z)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_features + n_features * n_clusters]) The transformed data.
      def transform(x)
        clustering(x) if @cluster_centers.nil?

        cluster_ids = assign_cluster_id(x)

        x = expand_feature(x) if fit_bias?

        n_samples, n_features = x.shape
        z = Numo::DFloat.zeros(n_samples, n_features * (1 + @params[:n_clusters]))
        z[true, 0...n_features] = 1.fdiv(Math.sqrt(@params[:reg_param_global])) * x
        @params[:n_clusters].times do |n|
          assigned_bits = cluster_ids.eq(n)
          z[assigned_bits.where, n_features * (n + 1)...n_features * (n + 2)] = x[assigned_bits.where, true]
        end

        z
      end

      private

      def linear_svc_params
        @params.reject { |key, _| CLUSTERED_SVC_BINARY_PARAMS.include?(key) }.merge(fit_bias: false)
      end

      def clustering(x)
        n_samples = x.shape[0]
        sub_rng = @rng.dup
        rand_id = Array.new(@params[:n_clusters]) { |_v| sub_rng.rand(0...n_samples) }
        @cluster_centers = x[rand_id, true].dup

        @params[:max_iter_kmeans].times do |_t|
          center_ids = assign_cluster_id(x)
          old_centers = @cluster_centers.dup
          @params[:n_clusters].times do |n|
            assigned_bits = center_ids.eq(n)
            @cluster_centers[n, true] = x[assigned_bits.where, true].mean(axis: 0) if assigned_bits.count.positive?
          end
          error = Numo::NMath.sqrt(((old_centers - @cluster_centers)**2).sum(axis: 1)).mean
          break if error <= @params[:tol_kmeans]
        end
      end

      def assign_cluster_id(x)
        distance_matrix = ::Rumale::PairwiseMetric.euclidean_distance(x, @cluster_centers)
        distance_matrix.min_index(axis: 1) - Numo::Int32[*0.step(distance_matrix.size - 1, @cluster_centers.shape[0])]
      end

      def expand_feature(x)
        n_samples = x.shape[0]
        Numo::NArray.hstack([x, Numo::DFloat.ones([n_samples, 1]) * @params[:bias_scale]])
      end

      def fit_bias?
        return false if @params[:fit_bias].nil? || @params[:fit_bias] == false

        true
      end

      CLUSTERED_SVC_BINARY_PARAMS = %i[n_clusters reg_param_global max_iter_kmeans tol_kmeans].freeze

      private_constant :CLUSTERED_SVC_BINARY_PARAMS
    end
  end
end
