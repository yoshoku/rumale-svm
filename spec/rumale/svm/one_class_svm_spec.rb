# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::OneClassSVM do
  let(:dataset) { two_blobs }
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:pos_id) { dataset[1].eq(1).where }
  let(:neg_id) { dataset[1].ne(1).where }
  let(:x_pos) { dataset[0][pos_id, true] }
  let(:y_pos) { dataset[1][pos_id] }
  let(:x_neg) { dataset[0][neg_id, true] }
  let(:y_neg) { dataset[1][neg_id] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_neg_samples) { x_neg.shape[0] }
  let(:nu) { n_neg_samples.fdiv(n_samples) }
  let(:gamma) { 1.0 }
  let(:coef0) { 0.0 }
  let(:ocsvm) { described_class.new(nu: nu, kernel: kernel, gamma: gamma, coef0: coef0, degree: 2, random_seed: 1) }
  let(:copied) { Marshal.load(Marshal.dump(ocsvm)) }

  shared_examples 'distribution estimation task' do
    let(:dfs) { ocsvm.decision_function(x) }
    let(:predicted) { ocsvm.predict(x) }
    let(:probs) { ocsvm.predict_proba(x) }
    let(:n_sv) { ocsvm.n_support }

    before { ocsvm.fit(x_pos) }

    it 'evaluates performance', :aggregate_failures do
      expect(ocsvm.support.class).to eq(Numo::Int32)
      expect(ocsvm.support.ndim).to eq(1)
      expect(ocsvm.support.shape[0]).to eq(n_sv)
      expect(ocsvm.support_vectors.class).to eq(Numo::DFloat)
      expect(ocsvm.support_vectors.ndim).to eq(2)
      expect(ocsvm.support_vectors.shape[0]).to eq(n_sv)
      expect(ocsvm.support_vectors.shape[1]).to eq(n_features)
      expect(ocsvm.n_support.class).to eq(Integer)
      expect(ocsvm.duel_coef.class).to eq(Numo::DFloat)
      expect(ocsvm.duel_coef.ndim).to eq(2)
      expect(ocsvm.duel_coef.shape[0]).to eq(1)
      expect(ocsvm.duel_coef.shape[1]).to eq(n_sv)
      expect(ocsvm.intercept.class).to eq(Numo::DFloat)
      expect(ocsvm.intercept.ndim).to eq(1)
      expect(ocsvm.intercept.shape[0]).to eq(1)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.eq(1).count).to be >= 800
      expect(predicted.eq(-1).count).to be < 200
    end

    it 'calculates decision function values', :aggregate_failures do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(1)
      expect(dfs.shape[0]).to eq(n_samples)
      expect(dfs.ge(0).count).to eq(predicted.eq(1).count)
    end

    it 'estimates probabilities', :aggregate_failures do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(2)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.instance_variable_get(:@model)).to eq(ocsvm.instance_variable_get(:@model))
      expect(copied.params).to eq(ocsvm.params)
    end
  end

  context 'when kernel is "rbf"' do
    let(:kernel) { 'rbf' }
    let(:gamma) { 10.0 }

    it_behaves_like 'distribution estimation task'
  end

  context 'when kernel is "poly"' do
    let(:kernel) { 'poly' }

    it_behaves_like 'distribution estimation task'
  end

  context 'when kernel is "linear"' do
    let(:kernel) { 'linear' }

    it_behaves_like 'distribution estimation task'
  end

  context 'when kernel is "sigmoid"' do
    let(:kernel) { 'sigmoid' }
    let(:coef0) { 1.0 }

    it_behaves_like 'distribution estimation task'
  end

  context 'when kernel is "precomputed"' do
    let(:kernel) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.linear_kernel(dataset[0]) }
    let(:n_features) { 2 }

    it_behaves_like 'distribution estimation task'
  end

  context 'when called predict method before training with fit method' do
    let(:kernel) { 'linear' }

    it 'raises Runtime error', :aggregate_failures do
      expect { ocsvm.predict(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::OneClassSVM#predict expects to be called after training the model with the fit method.'
      )
      expect { ocsvm.decision_function(x) }.to raise_error(
        RuntimeError, 'Rumale::SVM::OneClassSVM#decision_function expects to be called after training the model with the fit method.'
      )
    end
  end
end
