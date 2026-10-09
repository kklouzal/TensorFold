"""Actual parameter-input and geometry-plan dataflow without MLX execution.

Opaque providers verify current descriptors reach exact original transforms,
captured-input dictionaries and projections. Native compilation/numerics and
end-to-end performance remain separate required Apple gates.
"""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'src/tensorfold/kernels/nemotron/lightning/v1/kernels.py').read_bytes())
OWNER = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'FusedDecode')
WANTED = {'_block', '_policy_value', '_mamba_parameters', '_set_block_parameters'}


class Array:
    def __init__(self, value, shape=(8,), dtype='F32'):
        self.value, self.shape, self.dtype = value, shape, dtype
    def __getitem__(self, key):
        return Array(('slice', self.value), self.shape, self.dtype)
    @property
    def T(self):
        return Array(('transpose', self.value), self.shape, self.dtype)
    def astype(self, dtype):
        return Array(self.value, self.shape, dtype)
    def __eq__(self, other):
        raise AssertionError('device-array comparison/synchronization is forbidden')


class Projection:
    def __init__(self, value):
        self.bits, self.group_size, self.mode = 4, 64, 'affine'
        self.weight = Array(value, (8, 8), 'U32')
        self.scales, self.biases = Array(value + '-s', (8, 1)), Array(value + '-b', (8, 1))


class Mixer(dict):
    def __init__(self):
        super().__init__()
        self.projection = Projection('weight1')
        self.conv1d = {'weight': Array('conv1', (4, 8, 1))}
        self.conv1d = SimpleNamespace(weight=self.conv1d['weight'])
        self.conv_dim = 8
        self.A_log, self.D, self.dt_bias = Array('A1'), Array('D1'), Array('dt1')
        self.gate = SimpleNamespace(e_score_correction_bias=Array('gate1'))
    @property
    def state(self):
        return self
    def named_modules(self):
        return [('projection', self.projection)]


class Conv(dict):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = weight
        if bias is not None:
            self['bias'] = self.bias = bias


def fixture():
    compiled, zeros = [], []
    def compile_function(function, *, inputs):
        compiled.append((function, inputs))
        return function
    def zero(shape):
        result = Array('zero', shape)
        zeros.append(result)
        return result
    namespace = {'mx': SimpleNamespace(array=Array, float32='F32', compile=compile_function, zeros=zero)}
    methods = [node for node in OWNER.body if isinstance(node, ast.FunctionDef) and node.name in WANTED]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    driver = ast.ClassDef(name='Methods', bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, driver], type_ignores=[])),
                 '<actual-Nemotron-parameter-owners>', 'exec'), namespace)
    class Driver(namespace['Methods']):
        def __init__(self):
            mixer = Mixer()
            mixer.conv1d = Conv(mixer.conv1d.weight)
            self.layers = [SimpleNamespace(mixer=mixer)]
            self.lane_xs = False
            self._compiled_blocks, self._zero_conv_bias, self.mamba, self.gate_bias = {}, {}, {}, {}
            self.eps, self.limits, self.scaling = Array('eps'), Array('limits'), Array('scaling')
        def _mamba_block(self, index, current):
            return lambda: current
        def _moe_block(self, index, current):
            return lambda: current
    return Driver(), compiled, zeros


class Current(unittest.TestCase):
    def test_actual_mamba_block_reads_epsilon_from_replaced_captured_container_slot(self):
        method = next(node for node in OWNER.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_mamba_block')
        observed = []
        inside, outside = object(), object()
        namespace = {'group_norm': lambda x, weight, eps, group: observed.append(eps) or x,
                     'mamba_step': lambda *args, **kwargs: (object(), 'conv-row', 'ssm-row')}
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                     '<actual-Mamba-captured-slot>', 'exec'), namespace)
        mixer = SimpleNamespace(in_proj=lambda x: x, out_proj=lambda x: x,
                                norm=SimpleNamespace(weight=object(), group_size=8))
        owner = SimpleNamespace(layers=[SimpleNamespace(mixer=mixer)], eps=outside,
                                heads=4, head_dim=8, groups=1, state_dim=16,
                                _use_sums=lambda x, xs: x,
                                _add_norm=lambda h, y, norm, xs, *, eps:
                                (observed.append(eps), y, xs))
        current = {'mamba': (object(),) * 5, 'limits': object(), 'norm': object(), 'eps': outside}
        block = namespace['_mamba_block'](owner, 0, current)
        # Provider tree_fill replaces captured slots during tracing. The owner
        # attribute remains a different closure value; no SDK is simulated.
        current['eps'] = inside
        block(object(), object(), object(), object(), object())
        self.assertEqual(observed, [inside, inside])

    def test_current_same_object_mamba_descriptor_and_absent_bias_constant(self):
        owner, compiled, zeros = fixture()
        mixer = owner.layers[0].mixer
        first = owner._mamba_parameters(0, mixer)
        mixer.conv1d.weight.value = 'conv2'
        mixer.A_log.value = 'A2'
        second = owner._mamba_parameters(0, mixer)
        self.assertNotEqual(first[0].value, second[0].value)
        self.assertEqual(second[0].value, ('transpose', ('slice', 'conv2')))
        self.assertEqual(second[2].value, 'A2')
        self.assertEqual(len(zeros), 1)
        mixer.conv1d['bias'] = mixer.conv1d.bias = Array('present-bias')
        self.assertEqual(owner._mamba_parameters(0, mixer)[1].value, 'present-bias')

    def test_explicit_compile_inputs_update_without_recreating_same_geometry(self):
        owner, compiled, zeros = fixture()
        first_norm = Array('norm1')
        first = owner._block(0, 'M', first_norm)
        current = compiled[0][1]
        self.assertIs(current['module'], owner.layers[0].mixer.state)
        self.assertIs(current['norm'], first_norm)
        owner.layers[0].mixer.conv1d.weight.value = 'conv2'
        second_norm = Array('norm2')
        self.assertIs(owner._block(0, 'M', second_norm), first)
        self.assertEqual(len(compiled), 1)
        self.assertIs(current['norm'], second_norm)
        self.assertEqual(current['mamba'][0].value, ('transpose', ('slice', 'conv2')))
        self.assertIs(current['eps'], owner.eps)
        self.assertIs(current['limits'], owner.limits)
        self.assertIs(current['scaling'], owner.scaling)

    def test_same_object_gate_bias_is_a_current_explicit_input(self):
        owner, compiled, zeros = fixture()
        norm = Array('norm')
        first = owner._block(0, 'E', norm)
        owner.layers[0].mixer.gate.e_score_correction_bias.value = 'gate2'
        self.assertIs(owner._block(0, 'E', norm), first)
        self.assertEqual(compiled[0][1]['gate_bias'].value, 'gate2')
        self.assertIs(owner.gate_bias[0], compiled[0][1]['gate_bias'])

    def test_scalar_policy_or_module_identity_change_replaces_only_one_owned_plan(self):
        owner, compiled, zeros = fixture()
        norm = Array('norm')
        first = owner._block(0, 'E', norm)
        owner.layers[0].mixer.projection.bits = 5
        self.assertIsNot(owner._block(0, 'E', norm), first)
        mixer = Mixer()
        mixer.conv1d = Conv(mixer.conv1d.weight)
        owner.layers[0].mixer = mixer
        owner._block(0, 'E', norm)
        self.assertEqual(len(owner._compiled_blocks), 1)
        self.assertEqual(len(compiled), 3)
        self.assertIs(compiled[-1][1]['module'], mixer.state)

    def test_unknown_or_array_host_policy_uses_current_ordinary_work_no_array_equality(self):
        for value in (object(), Array('host-policy')):
            owner, compiled, zeros = fixture()
            owner.layers[0].mixer.projection.mode = value
            current = owner._block(0, 'E', Array('norm'))()
            self.assertEqual(current['gate_bias'].value, 'gate1')
            self.assertEqual(compiled, [])
            self.assertEqual(owner._compiled_blocks, {})

    def test_current_stack_refresh_has_one_constructor_and_startup_only_evaluation(self):
        helper = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == '_stack_linears')
        calls = []
        class Quantized:
            def __init__(self, *args, **kwargs):
                calls.append('construct')
                self.mode = 'affine'
            def parameters(self):
                return {'w': self.weight, 's': self.scales, 'b': self.biases}
        nn, mlx = ModuleType('mlx.nn'), ModuleType('mlx')
        nn.QuantizedLinear = Quantized
        mlx.nn = nn
        def concatenate(arrays, axis=0):
            return Array(tuple(array.value for array in arrays))
        ns = {'mx': SimpleNamespace(concatenate=concatenate, eval=lambda fields: calls.append('eval'))}
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, helper], type_ignores=[])),
                     '<actual-current-stack>', 'exec'), ns)
        linears = [Projection('q'), Projection('k'), Projection('v')]
        with patch.dict(sys.modules, {'mlx': mlx, 'mlx.nn': nn}):
            stacked, cuts = ns['_stack_linears'](linears)
            linears[1].scales.value = 'k-current-scales'
            refreshed, current_cuts = ns['_stack_linears'](linears, stacked)
        self.assertIs(refreshed, stacked)
        self.assertEqual(calls, ['construct', 'eval'])
        self.assertEqual((cuts, current_cuts), ([8, 16], [8, 16]))
        self.assertEqual(stacked.scales.value, ('q-s', 'k-current-scales', 'v-s'))


if __name__ == '__main__':
    unittest.main()
