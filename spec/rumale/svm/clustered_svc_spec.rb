# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::SVM::ClusteredSVC do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:n_clusters) { 16 }
  let(:svc) { described_class.new(n_clusters: n_clusters, fit_bias: true, random_seed: 42) }
  let(:dfs) { svc.decision_function(x) }
  let(:predicted) { svc.predict(x) }
  let(:z) { svc.transform(x) }
  let(:copied) { Marshal.load(Marshal.dump(svc)) }

  shared_examples 'multiclass classification task' do
    before { svc.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
      expect(svc.score(x, y)).to eq(1.0)
      expect(svc.cluster_centers.class).to eq(Numo::DFloat)
      expect(svc.cluster_centers.ndim).to eq(2)
      expect(svc.cluster_centers.shape[0]).to eq(n_clusters)
      expect(svc.cluster_centers.shape[1]).to eq(n_features)
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

    it 'transforms given features', :aggregate_failures do
      expect(z.class).to eq(Numo::DFloat)
      expect(z).to be_contiguous
      expect(z.ndim).to eq(2)
      expect(z.shape[0]).to eq(n_samples)
      expect(z.shape[1]).to eq((n_features + 1) + (n_features + 1) * n_clusters)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.params).to eq(svc.params)
      expect(copied.cluster_centers).to eq(svc.cluster_centers)
      expect(copied.score(x, y)).to eq(svc.score(x, y))
    end
  end

  shared_examples 'binary classification task' do
    before { svc.fit(x, y) }

    it 'evaluates classification performance', :aggregate_failures do
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
      expect(copied.cluster_centers).to eq(svc.cluster_centers)
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
      expect(svc.score(x_valid, y_valid)).to be >= 0.98
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
