# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::NuSVR do
  let(:dataset) { three_blobs }
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:kernel) { 'rbf' }
  let(:svr) { described_class.new(kernel: kernel, gamma: 0.1, degree: 2, random_seed: 1) }
  let(:copied) { Marshal.load(Marshal.dump(svr)) }

  shared_examples 'regression task' do
    let(:predicted) { svr.predict(x) }
    let(:n_sv) { svr.n_support }

    before { svr.fit(x, y) }

    it 'evaluates regression performance', :aggregate_failures do
      expect(svr.support.class).to eq(Numo::Int32)
      expect(svr.support.ndim).to eq(1)
      expect(svr.support.shape[0]).to eq(n_sv)
      expect(svr.support_vectors.class).to eq(Numo::DFloat)
      expect(svr.support_vectors.ndim).to eq(2)
      expect(svr.support_vectors.shape[0]).to eq(n_sv)
      expect(svr.support_vectors.shape[1]).to eq(n_features)
      expect(svr.n_support.class).to eq(Integer)
      expect(svr.duel_coef.class).to eq(Numo::DFloat)
      expect(svr.duel_coef.ndim).to eq(2)
      expect(svr.duel_coef.shape[0]).to eq(1)
      expect(svr.duel_coef.shape[1]).to eq(n_sv)
      expect(svr.intercept.class).to eq(Numo::DFloat)
      expect(svr.intercept.ndim).to eq(1)
      expect(svr.intercept.shape[0]).to eq(1)
      expect(svr.score(x, y)).to be >= 0.95
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@params)).to eq(svr.instance_variable_get(:@params))
      expect(copied.instance_variable_get(:@model)).to eq(svr.instance_variable_get(:@model))
      expect(copied.score(x, y)).to eq(svr.score(x, y))
    end
  end

  shared_examples 'hold out' do
    let(:split) { Rumale::ModelSelection::KFold.new(n_splits: 5, shuffle: true, random_seed: 1).split(x, y).first }
    let(:train_id) { split[0] }
    let(:valid_id) { split[1] }
    let(:x_train) { x[train_id, true] }
    let(:y_train) { y[train_id] }
    let(:x_valid) { x[valid_id, true] }
    let(:y_valid) { y[valid_id] }

    before { svr.fit(x_train, y_train) }

    it 'evaluates classification performance' do
      expect(svr.score(x_valid, y_valid)).to be >= 0.95
    end
  end

  context 'when kernel is "rbf"' do
    let(:kernel) { 'rbf' }
    let(:y) { Numo::NMath.sin(dataset[0].dot(Rumale::Utils.rand_normal([2, 1], Random.new(1))).flatten) }

    it_behaves_like 'regression task'
    it_behaves_like 'hold out'
  end

  context 'when kernel is "poly"' do
    let(:kernel) { 'poly' }
    let(:y) { dataset[0].dot(Rumale::Utils.rand_normal([2, 1], Random.new(1))).flatten**2 }

    it_behaves_like 'regression task'
  end

  context 'when kernel is "linear"' do
    let(:kernel) { 'linear' }
    let(:y) { dataset[0].dot(Rumale::Utils.rand_normal([2, 1], Random.new(1))).flatten }

    it_behaves_like 'regression task'
    it_behaves_like 'hold out'
  end

  context 'when kernel is "sigmoid"' do
    let(:kernel) { 'sigmoid' }
    let(:x) { (dataset[0] - dataset[0].mean(0)) / dataset[0].stddev(0) }
    let(:y) { x.dot(Rumale::Utils.rand_normal([2, 1], Random.new(1))).flatten**3 }

    it_behaves_like 'regression task'
  end

  context 'when kernel is "precomputed"' do
    let(:kernel) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.linear_kernel(dataset[0]) }
    let(:y) { dataset[0].dot(Rumale::Utils.rand_normal([2, 1], Random.new(1))).flatten }

    it_behaves_like 'regression task'
  end

  context 'when called predict method before training with fit method' do
    it 'raises Runtime error' do
      expect { svr.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::NuSVR#predict expects to be called after training the model with the fit method.'
      )
    end
  end
end
