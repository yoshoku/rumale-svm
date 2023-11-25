# frozen_string_literal: true

require 'simplecov'
SimpleCov.start

require 'bundler/setup'
require 'rumale/svm'
require 'rumale/utils'
require 'rumale/dataset'
require 'rumale/pairwise_metric'
require 'rumale/model_selection/stratified_k_fold'
require 'rumale/model_selection/k_fold'
require 'lbfgsb'
require 'numo/tiny_linalg'
Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)

def xor_data(n_samples = 1000)
  rng = Random.new(1)
  clstr_size = n_samples / 4
  a = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[-5, 5]
  b = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[5, -5]
  c = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[-5, -5]
  d = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[5, 5]
  x = Numo::NArray.vstack([a, b, c, d])
  y = Numo::NArray.concatenate([Numo::Int32.zeros(clstr_size * 2) + 1, Numo::Int32.zeros(clstr_size * 2) + 2])
  [x, y]
end

def three_blobs(n_samples = 900)
  rng = Random.new(1)
  clstr_size = n_samples / 3
  a = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[-5, -5]
  b = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[5, -5]
  c = Rumale::Utils.rand_normal([clstr_size, 2], rng, 0.0, 0.5) - Numo::DFloat[0, 5]
  x = Numo::NArray.vstack([a, b, c])
  y = Numo::NArray.concatenate([Numo::Int32.zeros(clstr_size) + 1,
                                Numo::Int32.zeros(clstr_size) + 2,
                                Numo::Int32.zeros(clstr_size) + 3])
  [x, y]
end

def two_blobs(n_samples = 1000)
  rng = Random.new(1)
  pos_clstr_size = (n_samples * 0.95).to_i
  neg_clstr_size = n_samples - pos_clstr_size
  a = Rumale::Utils.rand_normal([pos_clstr_size, 2], rng, 0.0, 0.1)
  b = Rumale::Utils.rand_normal([neg_clstr_size, 2], rng, 8.0, 0.1)
  x = Numo::NArray.vstack([a, b])
  y = Numo::NArray.concatenate([Numo::Int32.zeros(pos_clstr_size) + 1,
                                Numo::Int32.zeros(neg_clstr_size) - 1])
  [x, y]
end

def two_balls(n_samples = 1000)
  rng = Random.new(1)
  pos_clstr_size = (n_samples * 0.6).to_i
  neg_clstr_size = n_samples - pos_clstr_size
  a = Rumale::Utils.rand_normal([pos_clstr_size, 2], rng, 0.0, 0.5) + Numo::DFloat[-5, -8]
  b = Rumale::Utils.rand_normal([neg_clstr_size, 2], rng, 0.0, 0.5) + Numo::DFloat[8, 5]
  x = Numo::NArray.vstack([a, b])
  y = Numo::NArray.concatenate([Numo::Int32.zeros(pos_clstr_size) + 1,
                                Numo::Int32.zeros(neg_clstr_size) + 2])
  [x, y]
end

def two_moons(n_samples = 1000)
  Rumale::Dataset.make_moons(n_samples, random_seed: 1)
end

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = '.rspec_status'

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  config.define_derived_metadata do |meta|
    meta[:aggregate_failures] = true unless meta.key?(:aggregate_failures)
  end
end
