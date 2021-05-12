import torch
import numpy as np
import time
import numpy.matlib
from math import sqrt, cosh, pi

import logging

from lgn.g_lib import rotations as rot


def _gen_rot(angles, maxdim, device=torch.device('cpu'), dtype=torch.float64, cg_dict=None):

	# save the dictionary of Lorentz-D matrices
	D = {(k, n): rot.LorentzD((k, n), *angles, device=device, dtype=dtype, cg_dict=cg_dict) for k in range(maxdim) for n in range(maxdim)}
	# compute the Lorentz matrix in cartesian coordinates
	cartesian4=torch.tensor([[[1,0,0,0],[0,1/sqrt(2.),0,0],[0,0,0,1],[0,-1/sqrt(2.),0,0]],
                            [[0,0,0,0],[0,0,-1/sqrt(2.),0],[0,0,0,0],[0,0,-1/sqrt(2.),0]]], device=device, dtype=dtype)
	cartesian4H=torch.tensor([[[1,0,0,0],[0,1/sqrt(2.),0,0],[0,0,0,1],[0,-1/sqrt(2.),0,0]],
                            [[0,0,0,0],[0,0,1/sqrt(2.),0],[0,0,0,0],[0,0,1/sqrt(2.),0]]], device=device, dtype=dtype).permute(0,2,1)
	R = torch.stack((D[(1, 1)][0].matmul(cartesian4[0])-D[(1, 1)][1].matmul(cartesian4[1]), D[(1, 1)][0].matmul(cartesian4[1]) + D[(1, 1)][1].matmul(cartesian4[0])))
	R = cartesian4H[0].matmul(R[0]) - cartesian4H[1].matmul(R[1])

	return D, R

def covariance_test(encoder, decoder, data, test_type, beta_max=None, cg_dict=None):
	logging.info('Beginning covariance test...')

	if cg_dict is None:
		cg_dict = encoder.cg_dict

	device = encoder.device
	dtype = encoder.dtype
	# data['p4'] = data['p4'].to(device, dtype)
	data['p4']=torch.rand_like(data['p4'])

	covariance_test_result = dict()

	if test_type.lower() in ['boost', 'boosts']:
		if beta_max is None:
			beta_max = 10.
		alpha_range = np.arange(0, beta_max, step=beta_max/25.)
		gammas, boost_dev_output, boost_dev_internal = boost_equivariance(encoder, decoder, data, alpha_range, device, dtype, cg_dict)
		covariance_test_result['gammas'] = gammas
		covariance_test_result['boost_dev_output'] = boost_dev_output
		covariance_test_result['boost_dev_internal'] = boost_dev_internal

	elif test_type.lower() in ['rot', 'rotation', 'rotations']:
		thetas, rot_dev_output, rot_dev_internal = rot_equivariance(encoder, decoder, data, device, dtype, cg_dict)
		covariance_test_result['thetas'] = thetas
		covariance_test_result['rot_dev_output'] = rot_dev_output
		covariance_test_result['rot_dev_internal'] = rot_dev_internal

	else:
		raise ValueError(f"test_type must be one of 'boost' or 'rotation': {test_type}")

	return covariance_test_result

def boost_equivariance(encoder, decoder, data, alpha_range, device, dtype, cg_dict):
	gammas = list(cosh(x) for x in alpha_range)
	boost_input = []
	boost_output = []
	boost_input_nodes_all = []
	boost_output_nodes_all = []
	for alpha in alpha_range:
		# boost input
		Di, Ri = _gen_rot((0, 0, alpha*1j), encoder.maxdim, device=device, dtype=dtype, cg_dict=cg_dict)
		data_boost = data.copy()
		data_boost['p4'] = torch.einsum("...b, ba->...a", data['p4'], Ri)  # Boost input
		res_boost_input, internal_boost_input = get_output(encoder, decoder, data_boost, covariance_test=True)
		boost_input.append((res_boost_input))
		boost_input_nodes_all.append((internal_boost_input))

		# boost output
		res, internal = get_output(encoder, decoder, data, covariance_test=True)
		boost_res = rot.rotate_rep(res, 0, 0, alpha*1j, cg_dict=cg_dict)
		boost_internal = [rot.rotate_rep(internal[i], 0, 0, alpha*1j, cg_dict=cg_dict) for i in range(len(internal))]
		boost_output.append((boost_res))
		boost_output_nodes_all.append((boost_internal))

		dev_output, dev_internal = get_dev(boost_input, boost_output, boost_input_nodes_all, boost_output_nodes_all)

	return gammas, dev_output, dev_internal

def rot_equivariance(encoder, decoder, data, device, dtype, cg_dict):
	thetas = np.arange(0, 2 * pi, step=pi/10)
	rot_input = []
	rot_output = []
	rot_input_nodes_all = []
	rot_output_nodes_all = []
	for theta in thetas:
		# rotate input -> output
		Di, Ri = _gen_rot((0, theta, 0), encoder.maxdim, device=device, dtype=dtype, cg_dict=cg_dict)
		data_boost = data.copy()
		data_boost['p4'] = torch.einsum("...b, ba->...a", data['p4'], Ri)  # Boost input
		res_rot_input, internal_rot_input = get_output(encoder, decoder, data_boost, covariance_test=True)
		rot_input.append((res_rot_input))
		rot_input_nodes_all.append((internal_rot_input))

		# input -> rotate output
		res, internal = get_output(encoder, decoder, data, covariance_test=True)
		rot_res = rot.rotate_rep(res, 0, theta, 0, cg_dict=cg_dict)
		rot_internal = [rot.rotate_rep(internal[i], 0, theta, 0, cg_dict=cg_dict) for i in range(len(internal))]
		rot_output.append((rot_res))
		rot_output_nodes_all.append((rot_internal))

		dev_output, dev_internal = get_dev(rot_input, rot_output, rot_input_nodes_all, rot_output_nodes_all)

	return thetas, dev_output, dev_internal

def get_dev(transform_input, transform_output, transform_input_nodes_all, transform_output_nodes_all):
	# Output equivariance
	dev_output = [{weight: ((transform_input[i][weight] - transform_output[i][weight]) / (transform_output[i][weight])).abs().max() for weight in [(0,0), (1,1)]} for i in range(len(transform_output))]
	# Equivariance of all internal features
	dev_internal = [[{weight: ((transform_input_nodes_all[i][j][weight] - transform_output_nodes_all[i][j][weight]) / transform_output_nodes_all[i][j][weight]).abs().max() for weight in [(0,0), (1,1)]} for j in range(len(transform_output_nodes_all[i]))] for i in range(len(transform_output_nodes_all))]
	return dev_output, dev_internal

def lgn_tests(encoder, decoder, dataloader, args, epoch, beta_max=None, tests=['covariance','permutation','batch'], cg_dict=None):

	t0 = time.time()

	logging.info("Testing network for symmetries:")
	encoder.eval()
	decoder.eval()

	lgn_test_results = dict()

	data = next(iter(dataloader))
	if 'covariance' in tests:
		logging.info('Boost equivariance test begins...')
		boost_results = covariance_test(encoder, decoder, data, test_type='boost', cg_dict=cg_dict, beta_max=beta_max)
		logging.info('Boost equivariance test completed!')

		logging.info('Rotation equivariance test begins...')
		lgn_test_results.update(boost_results)
		rotation_results = covariance_test(encoder, decoder, data, test_type='rotation', cg_dict=cg_dict)
		logging.info('Rotation equivariance test completed!')

		lgn_test_results.update(rotation_results)
	# if 'permutation' in tests:
	# 	permutation_results = permutation_test(encoder, decoder, data)
	# 	lgn_test_results['permutation_invariance'] = permutation_results
	# if 'batch' in tests:
	# 	batch_results = batch_test(encoder, decoder, data)
	# 	lgn_test_results['batch_invariance'] = batch_results

	logging.info('Test complete!')
	dt = time.time() - t0
	print("Time it took testing equivariance of epoch", epoch+1, "is:", round(dt/60,2), "min")

	return lgn_test_results


def get_output(encoder, decoder, data, covariance_test=False):
	"""
	Get output and all internal features from the autoencoder.
	"""
	latent_features, encoder_nodes_all = encoder(data, covariance_test=covariance_test)
	generated_features, nodes_all = decoder(latent_features, covariance_test=covariance_test, nodes_all=encoder_nodes_all)
	return generated_features, nodes_all