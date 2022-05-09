# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::LogisticRegression do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:penalty) { 'l2' }
  let(:dual) { false }
  let(:fit_bias) { true }
  let(:logit) { described_class.new(penalty: penalty, dual: dual, fit_bias: fit_bias, random_seed: 1) }
  let(:dfs) { logit.decision_function(x) }
  let(:predicted) { logit.predict(x) }
  let(:probs) { logit.predict_proba(x) }
  let(:predict_pr) { Numo::Int32[*Array.new(n_samples) { |n| probs[n, true].max_index }] + 1 }
  let(:copied) { Marshal.load(Marshal.dump(logit)) }

  shared_examples 'multiclass classification task' do
    before { logit.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
      expect(logit.weight_vec.class).to eq(Numo::DFloat)
      expect(logit.weight_vec.ndim).to eq(2)
      expect(logit.weight_vec.shape[0]).to eq(n_classes)
      expect(logit.weight_vec.shape[1]).to eq(n_features)
      expect(logit.bias_term.class).to eq(Numo::DFloat)
      expect(logit.bias_term.ndim).to eq(1)
      expect(logit.bias_term.shape[0]).to eq(n_classes)
      expect(logit.score(x, y)).to eq(1.0)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'calculates decision function values', :aggregate_failures do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(2)
      expect(dfs.shape[0]).to eq(n_samples)
      expect(dfs.shape[1]).to eq(n_classes)
    end

    it 'predicts class probabilities', :aggregate_failures do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predict_pr).to eq(y)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@model)).to eq(logit.instance_variable_get(:@model))
      expect(copied.params).to eq(logit.params)
      expect(copied.weight_vec).to eq(logit.weight_vec)
      expect(copied.bias_term).to eq(logit.bias_term)
      expect(copied.score(x, y)).to eq(logit.score(x, y))
    end
  end

  shared_examples 'binary classification task' do
    before { logit.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
      expect(logit.weight_vec.class).to eq(Numo::DFloat)
      expect(logit.weight_vec.ndim).to eq(1)
      expect(logit.weight_vec.shape[0]).to eq(n_features)
      expect(logit.bias_term.class).to eq(Float)
      expect(logit.score(x, y)).to eq(1.0)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'calculates decision function values', :aggregate_failures do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(1)
      expect(dfs.shape[0]).to eq(n_samples)
    end

    it 'predicts class probabilities', :aggregate_failures do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predict_pr).to eq(y)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@params)).to eq(logit.instance_variable_get(:@params))
      expect(copied.instance_variable_get(:@model)).to eq(logit.instance_variable_get(:@model))
      expect(copied.weight_vec).to eq(logit.weight_vec)
      expect(copied.bias_term).to eq(logit.bias_term)
      expect(copied.score(x, y)).to eq(logit.score(x, y))
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

    before { logit.fit(x_train, y_train) }

    it 'evaluates classification performance' do
      expect(logit.score(x_valid, y_valid)).to be >= 0.95
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

    it 'raises Runtime error', :aggregate_failures do
      expect { logit.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LogisticRegression#predict expects to be called after training the model with the fit method.'
      )
      expect { logit.predict_proba(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LogisticRegression#predict_proba expects to be called after training the model with the fit method.'
      )
      expect { logit.decision_function(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LogisticRegression#decision_function expects to be called after training the model with the fit method.'
      )
    end
  end
end
