# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::LinearSVC do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:penalty) { 'l2' }
  let(:loss) { 'squared_hinge' }
  let(:dual) { false }
  let(:fit_bias) { true }
  let(:svc) { described_class.new(penalty: penalty, loss: loss, dual: dual, fit_bias: fit_bias, probability: true, random_seed: 1) }
  let(:dfs) { svc.decision_function(x) }
  let(:predicted) { svc.predict(x) }
  let(:probs) { svc.predict_proba(x) }
  let(:predict_pr) { Numo::Int32[*Array.new(n_samples) { |n| probs[n, true].max_index }] + 1 }
  let(:copied) { Marshal.load(Marshal.dump(svc)) }

  shared_examples 'multiclass classification task' do
    before { svc.fit(x, y) }

    it 'evaluates classification performance' do
      expect(svc.weight_vec.class).to eq(Numo::DFloat)
      expect(svc.weight_vec.ndim).to eq(2)
      expect(svc.weight_vec.shape[0]).to eq(n_classes)
      expect(svc.weight_vec.shape[1]).to eq(n_features)
      expect(svc.bias_term.class).to eq(Numo::DFloat)
      expect(svc.bias_term.ndim).to eq(1)
      expect(svc.bias_term.shape[0]).to eq(n_classes)
      expect(svc.score(x, y)).to eq(1.0)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'calculates decision function values' do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(2)
      expect(dfs.shape[0]).to eq(n_samples)
      expect(dfs.shape[1]).to eq(n_classes)
    end

    it 'predicts class probabilities' do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predict_pr).to eq(y)
    end

    it 'dumps and restores itself using Marshal module.' do
      expect(copied.instance_variable_get(:@model)).to eq(svc.instance_variable_get(:@model))
      expect(copied.instance_variable_get(:@prob_param)).to eq(svc.instance_variable_get(:@prob_param))
      expect(copied.params).to eq(svc.params)
      expect(copied.weight_vec).to eq(svc.weight_vec)
      expect(copied.bias_term).to eq(svc.bias_term)
      expect(copied.score(x, y)).to eq(svc.score(x, y))
    end
  end

  shared_examples 'binary classification task' do
    before { svc.fit(x, y) }

    it 'evaluates classification performance' do
      expect(svc.weight_vec.class).to eq(Numo::DFloat)
      expect(svc.weight_vec.ndim).to eq(1)
      expect(svc.weight_vec.shape[0]).to eq(n_features)
      expect(svc.bias_term.class).to eq(Float)
      expect(svc.score(x, y)).to eq(1.0)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'calculates decision function values' do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(1)
      expect(dfs.shape[0]).to eq(n_samples)
    end

    it 'predicts class probabilities' do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predict_pr).to eq(y)
    end

    it 'dumps and restores itself using Marshal module.' do
      expect(copied.instance_variable_get(:@params)).to eq(svc.instance_variable_get(:@params))
      expect(copied.instance_variable_get(:@model)).to eq(svc.instance_variable_get(:@model))
      expect(copied.weight_vec).to eq(svc.weight_vec)
      expect(copied.bias_term).to eq(svc.bias_term)
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

  context 'when given multi-class dataset' do
    let(:dataset) { three_blobs }

    it_behaves_like 'multiclass classification task'
    it_behaves_like 'hold out'

    context 'when selected dual solver' do
      let(:dual) { true }

      it_behaves_like 'multiclass classification task'
      it_behaves_like 'hold out'
    end

    context 'when selected hinge loss' do
      let(:loss) { 'hinge' }

      it_behaves_like 'multiclass classification task'
      it_behaves_like 'hold out'
    end

    context 'when selected l1 penalty' do
      let(:penalty) { 'l1' }

      it_behaves_like 'multiclass classification task'
      it_behaves_like 'hold out'
    end
  end

  context 'when given binary class dataset' do
    let(:dataset) { two_balls }

    it_behaves_like 'binary classification task'
    it_behaves_like 'hold out'
  end

  context 'when called predict method before training with fit method' do
    let(:dataset) { two_balls }

    it 'raises Runtime error' do
      expect { svc.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LinearSVC#predict expects to be called after training the model with the fit method.'
      )
      expect { svc.predict_proba(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LinearSVC#predict_proba expects to be called after training the model with the fit method.'
      )
      expect { svc.decision_function(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LinearSVC#decision_function expects to be called after training the model with the fit method.'
      )
    end
  end
end
