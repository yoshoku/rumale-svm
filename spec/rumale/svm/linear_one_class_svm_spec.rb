# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::LinearOneClassSVM do
  let(:x_pos) { Rumale::Utils.rand_normal([980, 2], Random.new(1), 8.0, 1.0) }
  let(:x_neg) { Rumale::Utils.rand_normal([20, 2], Random.new(1), 0.0, 0.1) }
  let(:x) { Numo::NArray.vstack([x_pos, x_neg]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_neg_samples) { x_neg.shape[0] }
  let(:nu) { 0.01 }
  let(:ocsvm) { described_class.new(nu: nu, random_seed: 1) }
  let(:copied) { Marshal.load(Marshal.dump(ocsvm)) }

  describe 'outlier detection task' do
    let(:dfs) { ocsvm.decision_function(x) }
    let(:predicted) { ocsvm.predict(x) }

    before { ocsvm.fit(x_pos) }

    it 'evaluates performance', :aggregate_failures do
      expect(ocsvm.weight_vec.class).to eq(Numo::DFloat)
      expect(ocsvm.weight_vec.ndim).to eq(1)
      expect(ocsvm.weight_vec.shape[0]).to eq(n_features)
      expect(ocsvm.bias_term.class).to eq(Float)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.eq(1).count).to be >= 970
      expect(predicted.eq(-1).count).to be <= 30
    end

    it 'calculates decision function values', :aggregate_failures do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(1)
      expect(dfs.shape[0]).to eq(n_samples)
      expect(dfs.gt(0).count).to eq(predicted.eq(1).count)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@model)).to eq(ocsvm.instance_variable_get(:@model))
      expect(copied.params).to eq(ocsvm.params)
    end
  end

  context 'when called predict method before training with fit method' do
    it 'raises Runtime error', :aggregate_failures do
      expect { ocsvm.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LinearOneClassSVM#predict expects to be called after training the model with the fit method.'
      )
      expect { ocsvm.decision_function(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::LinearOneClassSVM#decision_function expects to be called after training the model with the fit method.'
      )
    end
  end
end
