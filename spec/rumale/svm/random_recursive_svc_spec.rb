# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::RandomRecursiveSVC do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:n_hidden_layers) { 2 }
  let(:svc) { described_class.new(n_hidden_layers: n_hidden_layers, beta: 0.5, reg_param: 1e-2, random_seed: 42) }
  let(:dfs) { svc.decision_function(x) }
  let(:predicted) { svc.predict(x) }
  let(:copied) { Marshal.load(Marshal.dump(svc)) }

  shared_examples 'multiclass classification task' do
    before { svc.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
      expect(svc.classifiers.size).to eq(n_hidden_layers + 1)
      expect(svc.random_matrices.size).to eq(n_hidden_layers)
      expect(svc.random_matrices[0].class).to eq(Numo::DFloat)
      expect(svc.random_matrices[0].ndim).to eq(2)
      expect(svc.random_matrices[0].shape[0]).to eq(n_classes)
      expect(svc.random_matrices[0].shape[1]).to eq(n_features)
      expect(svc.score(x, y)).to eq(1.0)
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

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.params).to eq(svc.params)
      expect(copied.random_matrices).to eq(svc.random_matrices)
      expect(copied.score(x, y)).to eq(svc.score(x, y))
    end
  end

  shared_examples 'binary classification task' do
    before { svc.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
      expect(svc.classifiers.size).to eq(n_hidden_layers + 1)
      expect(svc.random_matrices.size).to eq(n_hidden_layers)
      expect(svc.random_matrices[0].class).to eq(Numo::DFloat)
      expect(svc.random_matrices[0].ndim).to eq(2)
      expect(svc.random_matrices[0].shape[0]).to eq(n_classes)
      expect(svc.random_matrices[0].shape[1]).to eq(n_features)
      expect(svc.score(x, y)).to be >= 0.95
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
    end

    it 'calculates decision function values', :aggregate_failures do
      expect(dfs.class).to eq(Numo::DFloat)
      expect(dfs.ndim).to eq(1)
      expect(dfs.shape[0]).to eq(n_samples)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.params).to eq(svc.params)
      expect(copied.random_matrices).to eq(svc.random_matrices)
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
  end

  context 'when given binary class dataset' do
    let(:dataset) { two_moons }

    it_behaves_like 'binary classification task'
    it_behaves_like 'hold out'
  end
end
