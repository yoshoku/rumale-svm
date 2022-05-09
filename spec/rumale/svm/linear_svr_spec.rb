# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::LinearSVR do
  let(:dataset) { three_blobs }
  let(:x) { Rumale::PairwiseMetric.linear_kernel(dataset[0]) }
  let(:y) { dataset[0].dot(Rumale::Utils.rand_normal([2, 1], Random.new(1))).flatten }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:loss) { 'squared_epsilon_insensitive' }
  let(:dual) { false }
  let(:fit_bias) { true }
  let(:svr) { described_class.new(loss: loss, dual: dual, fit_bias: fit_bias, random_seed: 1) }
  let(:predicted) { svr.predict(x) }
  let(:copied) { Marshal.load(Marshal.dump(svr)) }

  shared_examples 'regression task' do
    before { svr.fit(x, y) }

    it 'evaluates regression performance', :aggregate_failures do
      expect(svr.weight_vec.class).to eq(Numo::DFloat)
      expect(svr.weight_vec.ndim).to eq(1)
      expect(svr.weight_vec.shape[0]).to eq(n_features)
      expect(svr.bias_term.class).to eq(Float)
      expect(svr.score(x, y)).to be >= 0.95
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@model)).to eq(svr.instance_variable_get(:@model))
      expect(copied.instance_variable_get(:@prob_param)).to eq(svr.instance_variable_get(:@prob_param))
      expect(copied.params).to eq(svr.params)
      expect(copied.weight_vec).to eq(svr.weight_vec)
      expect(copied.bias_term).to eq(svr.bias_term)
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

  context 'when selected squared epsilon insensitive loss' do
    let(:loss) { 'squared_epsilon_insensitive' }

    it_behaves_like 'regression task'
    it_behaves_like 'hold out'
  end

  context 'when selected epsilon insensitive loss' do
    let(:loss) { 'epsilon_insensitive' }

    it_behaves_like 'regression task'
    it_behaves_like 'hold out'
  end

  context 'when selected dual solver' do
    let(:dual) { true }

    it_behaves_like 'regression task'
    it_behaves_like 'hold out'
  end

  context 'when not fitting bias' do
    let(:bias_term) { false }

    it_behaves_like 'regression task'
    it_behaves_like 'hold out'
  end

  context 'when called predict method before training with fit method' do
    it 'raises Runtime error' do
      expect { svr.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LinearSVR#predict expects to be called after training the model with the fit method.'
      )
    end
  end
end
