"""v5 numeric generation must remain identical to frozen v4."""
import ast
import copy
import json
import unittest
from rank_validation import generation_v4 as v4
from rank_validation import generation_v5 as v5

class MetadataNormalize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            allowed={'public_profile_joint_v5':'public_profile_joint_v4','public_main_profile_v5':'public_main_profile_v4','profile_v5_':'profile_v4_'}
            if node.value in allowed:
                return ast.copy_location(ast.Constant(allowed[node.value]),node)
        return node
    def visit_Dict(self,node):
        self.generic_visit(node)
        for i,key in enumerate(node.keys):
            if isinstance(key,ast.Constant) and key.value=='version' and isinstance(node.values[i],ast.Constant) and node.values[i].value==5:
                node.values[i]=ast.Constant(4)
        return node

class GenerationV5Tests(unittest.TestCase):
    def test_every_function_ast_identical_except_explicit_version_metadata(self):
        parse=lambda module:{n.name:n for n in ast.parse(module.Path(module.__file__).read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef)}
        old,new=parse(v4),parse(v5)
        self.assertEqual(set(old),set(new))
        for name in old:
            a=ast.dump(old[name],include_attributes=False)
            b=ast.dump(MetadataNormalize().visit(copy.deepcopy(new[name])),include_attributes=False)
            self.assertEqual(a,b,name)

    def test_all_numeric_generation_protocol_fields_unchanged(self):
        old,new=v4.load_protocol(),v5.load_protocol()
        allowed={'protocol_version','development_seeds','confirmation_seeds','frozen_utc','change_basis','prefit_revision','base_v4_protocol_sha256','base_v4_generator_sha256','metric_design_uses_retrospective_feedback','metric_validation','metric_protocol_sha256','pre_generation_protocol_revision_sha256'}
        for key in set(old)|set(new):
            if key not in allowed:
                self.assertEqual(old.get(key),new.get(key),key)
        self.assertEqual(new['base_v4_protocol_sha256'],v4.PROTOCOL_SHA256)
        self.assertEqual(new['base_v4_generator_sha256'],v4._sha(v4.__file__))

    def test_only_two_unused_holdout_seeds_reserved(self):
        protocol=v5.load_protocol()
        self.assertEqual(protocol['development_seeds'],[])
        self.assertEqual(protocol['confirmation_seeds'],[2026100961,2026100962])
        for seed in protocol['confirmation_seeds']:
            self.assertEqual(v5._reserved_split(seed,protocol),('holdout','confirmation'))
            with self.assertRaises(ValueError): v5.generate_world(seed,'dev')
        for seed in [2026100941,2026100951,2026100952]:
            with self.assertRaises(ValueError): v5._reserved_split(seed,protocol)

    def test_metric_separate_hash_no_fit_and_retrospective_cases(self):
        import hashlib
        p=v5.load_protocol(); m=p['metric_validation']
        data=(json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode()
        self.assertEqual(hashlib.sha256(data).hexdigest(),p['metric_protocol_sha256'])
        self.assertFalse(m['coefficients_fitted'])
        self.assertEqual(m['retrospective_design_cases'],[2026100941,2026100951,2026100952])
        self.assertIn('average those two case scores equally',m['validation'])
        self.assertIn('1e-12',m['validation'])
        self.assertIn('stronger both-cases-all-six',m['secondary_validation'])
        self.assertFalse(m['official_scorer_or_scores_changed'])

if __name__=='__main__': unittest.main()
