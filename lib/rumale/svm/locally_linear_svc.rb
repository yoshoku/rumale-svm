# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/pairwise_metric'
require 'rumale/utils'
require 'rumale/validation'

module Rumale
  module SVM
    # LocallyLinearSVC is a class that implements Locally Linear Support Vector Classifier with the squared hinge loss.
    # This classifier requires Numo::Linalg (or Numo::TinyLinalg) and Lbfgsb gems,
    # but they are listed in the runtime dependencies of Rumale::SVM.
    # Therefore, you should install and load Numo::Linalg and Lbfgsb gems explicitly to use this classifier.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'lbfgsb'
    #   require 'rumale/svm'
    #
    #   estimator = Rumale::SVM::LocallyLinearSVC.new(reg_param: 1.0, n_anchors: 128)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Ladicky, L., and Torr, P H.S., "Locally Linear Support Vector Machines," Proc. ICML'11, pp. 985--992, 2011.
    class LocallyLinearSVC < Rumale::Base::Estimator
      include Rumale::Base::Classifier

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the anchor vectors.
      # @return [Numo::DFloat] (shape: [n_anchors, n_features])
      attr_reader :anchors

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_classes, n_anchors, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept).
      # @return [Numo::DFloat] (shape: [n_classes, n_anchors])
      attr_reader :bias_term

      # Create a new classifier with Locally Linear Support Vector Machine.
      #
      # @param reg_param [Float] The regularization parameter for weight vector.
      # @param reg_param_local [Float] The regularization parameter for local coordinate.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion for finding anchors with k-means algorithm.
      # @param n_anchors [Integer] The number of anchors.
      # @param n_neighbors [Integer] The number of neighbors.
      # @param fit_bias [Boolean] The flag indicating whether to fit bias term.
      # @param bias_scale [Float] The scale parameter for bias term.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, reg_param_local: 1e-4, max_iter: 100, tol: 1e-4,
                     n_anchors: 128, n_neighbors: 10, fit_bias: true, bias_scale: 1.0, random_seed: nil)
        raise 'LocallyLinearSVC requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        super()
        @params = {
          reg_param: reg_param,
          reg_param_local: reg_param_local,
          max_iter: max_iter,
          n_anchors: n_anchors,
          tol: tol,
          n_neighbors: n_neighbors,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [LocallyLinearSVC] The learned classifier itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_label_array(y)
        Rumale::Validation.check_sample_size(x, y)
        raise 'LocallyLinearSVC#fit requires Lbfgsb but that is not loaded' unless defined?(Lbfgsb)

        @classes = Numo::Int32[*y.to_a.uniq.sort]

        find_anchors(x)
        n_samples, n_features = x.shape
        @coeff = Numo::DFloat.zeros(n_samples, @params[:n_anchors])
        n_samples.times do |i|
          xi = x[i, true]
          @coeff[i, true] = local_coordinates(xi)
        end

        x = expand_feature(x) if fit_bias?

        if multiclass_problem?
          n_classes = @classes.size
          @weight_vec = Numo::DFloat.zeros(n_classes, @params[:n_anchors], n_features)
          @bias_term = Numo::DFloat.zeros(n_classes, @params[:n_anchors])
          n_classes.times do |n|
            bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
            w, b = partial_fit(x, bin_y)
            @weight_vec[n, true, true] = w
            @bias_term[n, true] = b
          end
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term = partial_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        x = Rumale::Validation.check_convert_sample_array(x)
        n_samples = x.shape[0]

        if multiclass_problem?
          n_classes = @classes.size
          df = Numo::DFloat.zeros(n_samples, n_classes)
          n_samples.times do |i|
            xi = x[i, true]
            coeff = local_coordinates(xi)
            n_classes.times do |j|
              df[i, j] = coeff.dot(@weight_vec[j, true, true]).dot(xi) + coeff.dot(@bias_term[j, true])
            end
          end
        else
          df = Numo::DFloat.zeros(n_samples)
          n_samples.times do |i|
            xi = x[i, true]
            coeff = local_coordinates(xi)
            df[i] = coeff.dot(@weight_vec).dot(xi) + coeff.dot(@bias_term)
          end
        end
        df
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = Rumale::Validation.check_convert_sample_array(x)
        n_samples = x.shape[0]

        if multiclass_problem?
          df = decision_function(x)
          predicted = Array.new(n_samples) { |n| @classes[df[n, true].max_index] }
        else
          df = decision_function(x).ge(0.0).to_a
          predicted = Array.new(n_samples) { |n| @classes[df[n]] }
        end
        Numo::Int32.asarray(predicted)
      end

      private

      def partial_fit(base_x, bin_y) # rubocop:disable Metrics/AbcSize
        fnc = proc do |w, x, y, coeff, reg_param|
          n_anchors = coeff.shape[1]
          n_samples, n_features = x.shape
          w = w.reshape(n_anchors, n_features)
          z = (coeff * x.dot(w.transpose)).sum(axis: 1)
          t = 1 - y * z
          indices = t.gt(0)
          grad = reg_param * w
          if indices.count.positive?
            sx = x[indices, true]
            sy = y[indices]
            sc = coeff[indices, true]
            sz = z[indices]
            grad += 2.fdiv(n_samples) * (sc.transpose * (sz - sy)).dot(sx)
          end
          loss = 0.5 * reg_param * w.dot(w.transpose).trace + (x.class.maximum(0, t)**2).sum.fdiv(n_samples)
          [loss, grad.reshape(n_anchors * n_features)]
        end

        n_features = base_x.shape[1]
        sub_rng = @rng.dup
        w_init = 2.0 * ::Rumale::Utils.rand_uniform(@params[:n_anchors] * n_features, sub_rng) - 1.0

        res = Lbfgsb.minimize(
          fnc: fnc, jcb: true, x_init: w_init, args: [base_x, bin_y, @coeff, @params[:reg_param]],
          maxiter: @params[:max_iter], factr: @params[:tol] / Lbfgsb::DBL_EPSILON,
          verbose: @params[:verbose] ? 1 : -1
        )

        w = res[:x].reshape(@params[:n_anchors], n_features)

        if fit_bias?
          [w[true, 0...-1].dup, w[true, -1].dup]
        else
          [w, Numo::DFloat.zeros(@params[:n_anchors])]
        end
      end

      def local_coordinates(xi)
        neighbor_ids = find_neighbors(xi)
        diff = @anchors[neighbor_ids, true] - xi
        gram_mat = diff.dot(diff.transpose)
        gram_mat[gram_mat.diag_indices] += @params[:reg_param_local].fdiv(@params[:n_neighbors]) * gram_mat.trace
        local_coeff = Numo::Linalg.solve(gram_mat, Numo::DFloat.ones(@params[:n_neighbors]))
        local_coeff /= local_coeff.sum # + 1e-8
        coeff = Numo::DFloat.zeros(@params[:n_anchors])
        coeff[neighbor_ids] = local_coeff
        coeff
      end

      def find_neighbors(xi)
        diff = @anchors - xi
        dist = (diff**2).sum(axis: 1)
        dist.sort_index.to_a[0...@params[:n_neighbors]]
      end

      def find_anchors(x)
        n_samples = x.shape[0]
        sub_rng = @rng.dup
        rand_id = Array.new(@params[:n_anchors]) { |_v| sub_rng.rand(0...n_samples) }
        @anchors = x[rand_id, true].dup

        @params[:max_iter].times do |_t|
          center_ids = assign_anchors(x)
          old_anchors = @anchors.dup
          @params[:n_anchors].times do |n|
            assigned_bits = center_ids.eq(n)
            @anchors[n, true] = x[assigned_bits.where, true].mean(axis: 0) if assigned_bits.count.positive?
          end
          error = Numo::NMath.sqrt(((old_anchors - @anchors)**2).sum(axis: 1)).mean
          break if error <= @params[:tol]
        end
      end

      def assign_anchors(x)
        distance_matrix = ::Rumale::PairwiseMetric.euclidean_distance(x, @anchors)
        distance_matrix.min_index(axis: 1) - Numo::Int32[*0.step(distance_matrix.size - 1, @anchors.shape[0])]
      end

      def fit_bias?
        @params[:fit_bias] == true
      end

      def expand_feature(x)
        n_samples = x.shape[0]
        Numo::NArray.hstack([x, Numo::DFloat.ones([n_samples, 1]) * @params[:bias_scale]])
      end

      def multiclass_problem?
        @classes.size > 2
      end
    end
  end
end
