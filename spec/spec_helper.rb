# frozen_string_literal: true

require 'simplecov'
require 'coveralls'

Coveralls.wear!

SimpleCov.formatter = SimpleCov::Formatter::MultiFormatter.new(
  [SimpleCov::Formatter::HTMLFormatter, Coveralls::SimpleCov::Formatter]
)
SimpleCov.start

require 'bundler/setup'
require 'rumale/svm'
require 'rumale/utils'
require 'rumale/pairwise_metric'
require 'rumale/model_selection/stratified_k_fold'

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

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = '.rspec_status'

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end
