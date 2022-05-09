# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::NuSVC do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:kernel) { 'rbf' }
  let(:svc) { described_class.new(kernel: kernel, gamma: 0.1, degree: 2, random_seed: 1) }
  let(:copied) { Marshal.load(Marshal.dump(svc)) }

  shared_examples 'classification task' do
    let(:dfs) { svc.decision_function(x) }
    let(:predicted) { svc.predict(x) }
    let(:probs) { svc.predict_proba(x) }
    let(:predict_pr) { Numo::Int32[*Array.new(n_samples) { |n| probs[n, true].max_index }] + 1 }
    let(:n_sv) { svc.n_support.sum }

    before { svc.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
      expect(svc.support.class).to eq(Numo::Int32)
      expect(svc.support.ndim).to eq(1)
      expect(svc.support.shape[0]).to eq(n_sv)
      expect(svc.support_vectors.class).to eq(Numo::DFloat)
      expect(svc.support_vectors.ndim).to eq(2)
      expect(svc.support_vectors.shape[0]).to eq(n_sv)
      expect(svc.support_vectors.shape[1]).to eq(n_features)
      expect(svc.n_support.class).to eq(Numo::Int32)
      expect(svc.n_support.ndim).to eq(1)
      expect(svc.n_support.shape[0]).to eq(n_classes)
      expect(svc.duel_coef.class).to eq(Numo::DFloat)
      expect(svc.duel_coef.ndim).to eq(2)
      expect(svc.duel_coef.shape[0]).to eq(n_classes - 1)
      expect(svc.duel_coef.shape[1]).to eq(n_sv)
      expect(svc.intercept.class).to eq(Numo::DFloat)
      expect(svc.intercept.ndim).to eq(1)
      expect(svc.intercept.shape[0]).to eq(n_classes * (n_classes - 1) / 2)
      expect(svc.prob_a.class).to eq(Numo::DFloat)
      expect(svc.prob_a.ndim).to eq(1)
      expect(svc.prob_a.shape[0]).to eq(n_classes * (n_classes - 1) / 2)
      expect(svc.prob_b.class).to eq(Numo::DFloat)
      expect(svc.prob_b.ndim).to eq(1)
      expect(svc.prob_b.shape[0]).to eq(n_classes * (n_classes - 1) / 2)
      expect(svc.score(x, y)).to eq(1.0)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'calculates decision function values', :aggregate_failures do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(2)
      expect(dfs.shape[0]).to eq(n_samples)
      expect(dfs.shape[1]).to eq(n_classes * (n_classes - 1) / 2)
    end

    it 'predicts class probabilities', :aggregate_failures do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predict_pr).to eq(y)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@params)).to eq(svc.instance_variable_get(:@params))
      expect(copied.instance_variable_get(:@model)).to eq(svc.instance_variable_get(:@model))
      expect(copied.score(x, y)).to eq(svc.score(x, y))
    end
  end

  shared_examples 'hold out' do
    let(:split) { Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1).split(x, y).first }
    let(:train_id) { split[0] }
    let(:valid_id) { split[1] }
    let(:x_train) { dataset[0][train_id, true] }
    let(:y_train) { dataset[1][train_id] }
    let(:x_valid) { dataset[0][valid_id, true] }
    let(:y_valid) { dataset[1][valid_id] }

    before { svc.fit(x_train, y_train) }

    it 'evaluates classification performance' do
      expect(svc.score(x_valid, y_valid)).to be >= 0.95
    end
  end

  context 'when kernel is "rbf"' do
    let(:kernel) { 'rbf' }
    let(:dataset) { xor_data }

    it_behaves_like 'classification task'
    it_behaves_like 'hold out'
  end

  context 'when kernel is "poly"' do
    let(:kernel) { 'poly' }
    let(:dataset) { xor_data }

    it_behaves_like 'classification task'
  end

  context 'when kernel is "linear"' do
    let(:kernel) { 'linear' }
    let(:dataset) { three_blobs }

    it_behaves_like 'classification task'
  end

  context 'when kernel is "sigmoid"' do
    let(:kernel) { 'sigmoid' }
    let(:dataset) { three_blobs }

    it_behaves_like 'classification task'
  end

  context 'when kernel is "precomputed"' do
    let(:kernel) { 'precomputed' }
    let(:dataset) { three_blobs }
    let(:x) { Rumale::PairwiseMetric.linear_kernel(dataset[0]) }

    it_behaves_like 'classification task'
  end

  context 'when called predict method before training with fit method' do
    let(:dataset) { three_blobs }

    it 'raises Runtime error', :aggregate_failures do
      expect { svc.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::NuSVC#predict expects to be called after training the model with the fit method.'
      )
      expect { svc.predict_proba(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::NuSVC#predict_proba expects to be called after training the model with the fit method.'
      )
      expect { svc.decision_function(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::NuSVC#decision_function expects to be called after training the model with the fit method.'
      )
    end
  end
end
