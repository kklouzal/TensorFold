"""Independent CPU payload sentinels and exhaustive-cut oracle; no Torch."""
import gc
import unittest
from unittest.mock import patch
from tensorfold.engine.prefix_blocks import PrefixBlocks, PrefixCapacity


class PrefixBlockTests(unittest.TestCase):
    def publish(self, store, tokens, start=0, stop=None, groups=('full','window','hidden'), tags=(), finish=True):
        stop = len(tokens) if stop is None else stop
        data = {g: tuple((g,t) for t in tokens[start:stop]) for g in groups}
        charges = {g:128+96*(stop-start) for g in groups}
        writer = store.begin(tokens,start,stop,charges,tags=tags)
        writer.commit(data)
        if finish:
            writer.finish()
        return writer,data,charges

    def rows(self, hit, group, start, stop):
        return tuple(x for data,offset,count in hit.spans(group,start,stop) for x in data[offset:offset+count])

    def test_multiblock_interior_shared_payload_identity(self):
        store = PrefixBlocks(2_000_000)
        tokens = tuple(range(1024))
        _,data,charges = self.publish(store,tokens)
        identity = store.identity(tokens[:317],{'full':None,'window':11})
        hit = store.match(tokens[:317],{'full':None,'window':11})
        self.assertEqual(hit.length,317)
        self.assertEqual(hit.spans('hidden',316,317),((data['hidden'],316,1),))
        self.assertEqual(self.rows(hit,'window',306,317),data['window'][306:317])
        hit.close()
        branch = tokens[:317]+tuple(range(10000,10016))
        _,_,suffix = self.publish(store,branch,317)
        self.assertEqual(store.stats['payload_bytes'],sum(charges.values())+sum(suffix.values()))
        self.assertEqual(store.identity(tokens[:317],{'full':None,'window':11}),identity)
        for prompt in (tokens[:330],branch,tokens[:317]):
            hit = store.match(prompt,{'full':None,'window':11})
            self.assertEqual(hit.length,len(prompt))
            self.assertEqual(self.rows(hit,'full',0,hit.length),tuple(('full',t) for t in prompt))
            hit.close()

    def test_a_b_a_reuses_shared_prefix(self):
        store = PrefixBlocks(500_000,block_tokens=4)
        a,b = (1,2,3,4,5,6),(1,2,3,4,50,60)
        self.publish(store,a)
        hit = store.match(b,{'full':None,'window':2})
        self.assertEqual(hit.length,4)
        hit.close()
        self.publish(store,b,4)
        for prompt in (a,b,a):
            hit = store.match(prompt,{'full':None,'window':2})
            self.assertEqual(self.rows(hit,'full',0,hit.length),tuple(('full',t) for t in prompt))
            hit.close()
        self.assertEqual(store.stats['payload_bytes'],3*(128+96*6)+3*(128+96*2))

    def test_future_media_preserves_earlier_equal_prefix(self):
        store = PrefixBlocks(500_000)
        source = tuple(range(64))
        self.publish(store,source,tags=((32,48,'source-image'),))
        hit = store.match(source[:16],{'full':None})
        self.assertEqual(hit.length,16)
        hit.close()
        hit = store.match(source[:32]+(999,),{'full':None},tags=((32,33,'other-image'),))
        self.assertEqual(hit.length,32)
        hit.close()

    def test_atomic_media_cut_refused_in_both_directions(self):
        tokens = tuple(range(32))
        store = PrefixBlocks(500_000)
        self.publish(store,tokens,tags=((5,24,'image'),))
        hit = store.match(tokens[:16],{'full':None})
        self.assertEqual(hit.length,5)
        hit.close()
        empty = PrefixBlocks(500_000)
        self.publish(empty,tokens)
        hit = empty.match(tokens[:16],{'full':None},tags=((5,16,'image'),))
        self.assertEqual(hit.length,5)
        hit.close()

    def test_window_eviction_preserves_independent_full_and_hidden(self):
        store = PrefixBlocks(500_000)
        tokens = tuple(range(128))
        _,_,charge = self.publish(store,tokens)
        hit = store.match(tokens,{'full':None,'window':8})
        hit.close()
        # Pin hidden: equal-cost full/window then resolve by componentkind;
        # this isolates independent residency, not the superseded window-first policy.
        hidden = store.match(tokens, {'hidden': None}, require_end_group=None)
        free = store.budget_bytes-store.stats['charged_bytes']
        temporary = store.reserve_temporary(free+8000)
        self.assertLess(store.stats['payload_bytes'],sum(charge.values()))
        self.assertIsNone(store.match(tokens,{'full':None,'window':8}))
        hit = store.match(tokens,{'full':None})
        self.assertEqual(hit.length,len(tokens))
        hit.close()
        temporary.close()
        hidden.close()

    def test_explicit_lease_retirement_and_clear(self):
        store = PrefixBlocks(500_000)
        tokens = tuple(range(128))
        writer,_,_ = self.publish(store,tokens,finish=False)
        hit = store.match(tokens,{'full':None,'window':8})
        with self.assertRaises(PrefixCapacity):
            store.reserve_temporary(store.budget_bytes)
        with self.assertRaises(RuntimeError):
            store.clear()
        writer.finish()
        gc.collect()
        self.assertEqual(store.stats['read_leases'],1)
        hit.close()
        temp = store.reserve_temporary(1000)
        with self.assertRaises(RuntimeError):
            store.clear()
        temp.close()
        store.clear()
        self.assertEqual(store.stats['payload_bytes'],0)

    def test_commit_failure_retains_primary_writer_until_retired(self):
        store = PrefixBlocks(100_000)
        writer = store.begin((1,2),0,2,{'full':512})
        with self.assertRaises(ValueError) as found:
            writer.commit({'other':object()})
        self.assertTrue(store.stats['poisoned'])
        self.assertEqual(store.stats['write_leases'],1)
        with self.assertRaises(RuntimeError) as refused:
            store.match((1,2),{'full':None})
        self.assertIs(refused.exception.__cause__,found.exception)
        writer.abort()
        store.clear()
        self.assertTrue(store.stats['poisoned'])

    def test_closed_metadata_refusal_before_effects(self):
        store = PrefixBlocks(100_000)
        for tags in (((0,1,object()),),((1,1,'bad'),),((0,2,'a'),(1,2,'b'))):
            with self.subTest(tags=tags):
                with self.assertRaises((TypeError,ValueError)):
                    store.begin((1,2),0,2,{'full':100},tags=tags)
        with self.assertRaises(ValueError):
            store.begin(tuple(range(1025)),0,1025,{'full':100})
        self.assertEqual(store.stats['write_leases'],0)

    def test_exhaustive_independent_cut_oracle(self):
        store = PrefixBlocks(2_000_000,block_tokens=4)
        sources = [tuple(range(12)),(0,1,2,3,50,51,52,53),(0,1,2,90,91,92)]
        for tokens in sources:
            self.publish(store,tokens)
        for prompt in [s[:end]+tail for s in sources for end in range(1,len(s)+1) for tail in ((),(999,))]:
            for window in (1,3,9):
                expected = max((cut for cut in range(1,len(prompt)+1) if any(source[:cut]==prompt[:cut] and len(source)>=cut for source in sources)),default=0)
                hit = store.match(prompt,{'full':None,'window':window})
                self.assertEqual(0 if hit is None else hit.length,expected)
                if hit:
                    self.assertEqual(self.rows(hit,'full',0,hit.length),tuple(('full',t) for t in prompt[:hit.length]))
                    hit.close()

    def test_long_continuation_survives_eight_short_chats_and_pressure(self):
        store = PrefixBlocks(24_000_000)
        long = tuple(range(40_000))
        for start in range(0,len(long),1024):
            self.publish(store,long,start,min(start+1024,len(long)))
        hit = store.match(long+(999_999,), {'full':None,'window':2048})
        self.assertEqual(hit.length,len(long))
        hit.close()
        for n in range(8):
            short = tuple(range(100_000+2048*n,101_024+2048*n))
            self.publish(store,short)
        hit = store.match(long+(999_999,), {'full':None,'window':2048})
        self.assertEqual(hit.length,len(long))
        hit.close()
        free = store.budget_bytes-store.stats['charged_bytes']
        pressure = store.reserve_temporary(free+6_000_000)
        hit = store.match(long+(999_999,), {'full':None,'window':2048})
        self.assertIsNotNone(hit)
        self.assertEqual(hit.length,len(long))
        hit.close()
        pressure.close()
        self.assertLessEqual(store.stats['charged_bytes'],store.budget_bytes)

    def test_begin_primary_and_abort_failure_are_both_preserved(self):
        store = PrefixBlocks(100_000)
        primary, cleanup = MemoryError('insert'), OSError('abort')
        with patch.object(store,'_insert',side_effect=primary), patch.object(store,'_discard',side_effect=cleanup):
            with self.assertRaises(MemoryError) as found:
                store.begin((1,2),0,2,{'full':100})
        self.assertIs(found.exception,primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertTrue(store.stats['poisoned'])
        # Unknown cleanup is not retried or debited again. No GPU producer
        # ran; this pure-CPU failed owner remains terminally contained.
        with self.assertRaises(RuntimeError) as refused:
            store._writers[0].abort()
        self.assertIs(refused.exception.__cause__,cleanup)
        with self.assertRaises(RuntimeError):
            store.clear()

    def test_long_cold_window_has_boundary_not_token_candidate_count(self):
        store = PrefixBlocks(20_000_000)
        tokens = tuple(range(100_000))
        for start in range(0,len(tokens),1024):
            self.publish(store,tokens,start,min(len(tokens),start+1024),groups=('full','hidden'))
        with patch.object(store,'_coverage',wraps=store._coverage) as calls:
            self.assertIsNone(store.match(tokens,{'full':None,'window':16}))
        self.assertLessEqual(calls.call_count,4*(100_000//64+3))

    def test_pin_range_reuses_independent_storage_without_full_state_hit(self):
        store = PrefixBlocks(500_000)
        tokens = tuple(range(128))
        _,data,_ = self.publish(store,tokens,groups=('full','hidden'))
        self.assertIsNone(store.match(tokens,{'full':None,'window':16}))
        before = store.stats['payload_bytes']
        hit = store.pin_range(tokens,'full',64,96)
        self.assertEqual(hit.length,96)
        self.assertEqual(hit.spans('full',64,96),((data['full'],64,32),))
        self.assertIsNone(store.pin_range(tokens,'window',64,96))
        self.publish(store,tokens,64,96,groups=('window',))
        self.assertEqual(store.stats['payload_bytes'],before+128+96*32)
        self.assertEqual(self.rows(hit,'full',64,96),data['full'][64:96])
        hit.close()

    def test_range_path_cache_scans_long_input_once_between_structural_changes(self):
        store = PrefixBlocks(50_000_000)
        tokens = tuple(range(100_000))
        for start in range(0,len(tokens),1024):
            self.publish(store,tokens,start,min(start+1024,len(tokens)),groups=('full','hidden'))
        # No structural mutation: independent group batch lookup should not
        # copy/rescan all100K IDs for every1024-row group request.
        store._drop_path()
        with patch.object(store,'_scan_path',wraps=store._scan_path) as scan:
            for start in range(0,len(tokens),1024):
                hit = store.pin_range(tokens,'full',start,min(start+1024,len(tokens)))
                self.assertIsNotNone(hit)
                hit.close()
        self.assertEqual(scan.call_count,1)

    def test_holes_interval_solver_matches_independent_per_row_oracle(self):
        import random
        rng = random.Random(7811)
        tokens = tuple(range(48))
        for scenario in range(12):
            store = PrefixBlocks(500_000,block_tokens=4)
            resident = {g:set() for g in ('full','window','hidden')}
            for start in range(0,48,6):
                groups = tuple(g for g in resident if rng.randrange(3)) or ('hidden',)
                self.publish(store,tokens,start,start+6,groups=groups)
                for g in groups:
                    resident[g].update(range(start,start+6))
            for end in range(1,49):
                for window in (1,5,11):
                    prompt = tokens[:end]
                    valid = [cut for cut in range(1,end+1)
                             if set(range(cut)) <= resident['full']
                             and set(range(max(0,cut-window),cut)) <= resident['window']
                             and (cut < end or cut-1 in resident['hidden'])]
                    hit = store.match(prompt,{'full':None,'window':window})
                    self.assertEqual(0 if hit is None else hit.length,max(valid,default=0),(scenario,end,window))
                    if hit:
                        hit.close()

    def test_close_releases_payload_without_gc_and_is_final(self):
        import weakref
        class Owner:
            pass
        store = PrefixBlocks(100_000)
        payload = Owner()
        weak = weakref.ref(payload)
        writer = store.begin((1,2),0,2,{'full':512})
        writer.commit({'full':payload})
        writer.finish()
        del payload
        self.assertIsNotNone(weak())
        store.close()
        self.assertIsNone(weak())
        self.assertTrue(store.stats['closed'])
        store.close()
        with self.assertRaises(RuntimeError):
            store.begin((1,2),0,2,{'full':512})

    def test_stats_reservations_and_bounded_component_eviction_history(self):
        store = PrefixBlocks(100_000)
        writer = store.begin((1,2),0,2,{'full':512})
        self.assertEqual(store.stats['payload_bytes'],0)
        self.assertEqual(store.stats['reserved_payload_bytes'],512)
        writer.commit({'full':(1,2)})
        self.assertEqual(store.stats['payload_bytes'],512)
        self.assertEqual(store.stats['reserved_payload_bytes'],0)
        writer.finish()
        for n in range(45):
            self.publish(store,(100+n,200+n),groups=('full',))
        history = store.stats['eviction_history']
        self.assertTrue(0 < len(history) <= 32)
        self.assertTrue(all(set(row)=={'component','start','stop','bytes','reason'} for row in history))
        self.assertGreater(store.stats['chunks'],0)
        self.assertNotIn('records',store.stats)
        self.assertEqual(store.stats['eviction_policy'],'dependent-prefix-work-aging-bytes-tiebreak-v1')
        self.assertGreater(store.stats['indexed_unique_tokens'],0)
        self.assertLessEqual(store.stats['charged_bytes'],store.budget_bytes)

    def test_shared_root_full_group_outlives_cold_unrelated_leaf(self):
        store = PrefixBlocks(2_000_000)
        root = tuple(range(1024))
        writer,_,_ = self.publish(store,root,groups=('full',))
        branch_a = root+tuple(range(2000,3024))
        branch_b = root+tuple(range(4000,5024))
        self.publish(store,branch_a,1024,groups=('full',))
        self.publish(store,branch_b,1024,groups=('full',))
        cold = tuple(range(10000,11024))
        self.publish(store,cold,groups=('full',))
        free = store.budget_bytes-store.stats['charged_bytes']
        pressure = store.reserve_temporary(free+60_000)
        # More pressure without first touching/reheating the shared root:
        # dependency value must prefer an actual branch leaf over root data.
        free = store.budget_bytes-store.stats['charged_bytes']
        more = store.reserve_temporary(free+180_000)
        root_hit = store.pin_range(branch_a,'full',0,1024)
        self.assertIsNotNone(root_hit)
        self.assertIsNone(store.pin_range(cold,'full',0,1024))
        root_hit.close()
        more.close()
        pressure.close()
        self.assertIsNone(writer._record)

    def test_closed_identity_dag_is_bounded_and_bool_is_not_integer(self):
        store = PrefixBlocks(100_000)
        a,b = 'leaf','leaf'
        for _ in range(24):
            a,b = (a,a),(b,b)
        tokens = tuple(range(8))
        self.publish(store,tokens,tags=((0,4,a),))
        hit = store.match(tokens,{'full':None},tags=((0,4,b),))
        self.assertEqual(hit.length,8)
        hit.close()
        distinct = PrefixBlocks(100_000)
        self.publish(distinct,tokens,tags=((1,4,False),))
        hit = distinct.match(tokens,{'full':None},tags=((1,4,0),))
        self.assertEqual(hit.length,1)
        hit.close()

    def test_explicit_retired_host_metadata_credits_survive_data_clear(self):
        store = PrefixBlocks(200_000)
        empty = store.stats['charged_bytes']
        a,b = store.reserve_temporary(101),store.reserve_temporary(202)
        self.publish(store,(1,2),groups=('full',))
        with self.assertRaises(RuntimeError):
            store.clear()
        with self.assertRaises(RuntimeError):
            store.clear(preserve_temporary=(a,))
        with self.assertRaises(RuntimeError):
            store.clear(preserve_temporary=(a,a))
        other = PrefixBlocks(100_000)
        unrelated = other.reserve_temporary(101)
        with self.assertRaises(RuntimeError):
            store.clear(preserve_temporary=(a,unrelated))
        unrelated.close()
        writer,_,_ = self.publish(store,(1,2),groups=('full',),finish=False)
        with self.assertRaises(RuntimeError):
            store.clear(preserve_temporary=(a,b))
        writer.finish()
        reader = store.match((1,2),{'full':None},require_end_group=None)
        with self.assertRaises(RuntimeError):
            store.clear(preserve_temporary=(a,b))
        reader.close()
        store.clear(preserve_temporary=(a,b))
        self.assertEqual(store.stats['payload_bytes'],0)
        self.assertEqual(store.stats['temporary_bytes'],303)
        self.assertEqual(store.stats['charged_bytes'],empty+303+2*512)
        self.publish(store,(3,4),groups=('full',))
        with self.assertRaises(RuntimeError):
            store.close()
        a.close()
        b.close()
        store.close()
        self.assertTrue(store.stats['closed'])

    def test_writer_registration_oom_rolls_back_unescaped_no_producer_credit(self):
        for after in (False,True):
            with self.subTest(after_append=after):
                primary = MemoryError('writer journal')
                class Broken(list):
                    def append(self,value):
                        if after:
                            super().append(value)
                        raise primary
                store = PrefixBlocks(100_000)
                before = store.stats['charged_bytes']
                store._writers = Broken()
                with self.assertRaises(MemoryError) as found:
                    store.begin((1,2),0,2,{'full':1000})
                self.assertIs(found.exception,primary)
                self.assertEqual(store.stats['charged_bytes'],before)
                self.assertEqual(store.stats['write_leases'],0)
                self.assertEqual(store.stats['reserved_payload_bytes'],0)
                self.assertFalse(store.stats['poisoned'])
                store._writers = []
                self.publish(store,(3,4),groups=('full',))
                store.close()

    def test_temporary_registration_oom_has_no_unowned_credit(self):
        for after in (False,True):
            with self.subTest(after_append=after):
                primary = MemoryError('temporary journal')
                class Broken(list):
                    def append(self,value):
                        if after:
                            super().append(value)
                        raise primary
                store = PrefixBlocks(100_000)
                before = store.stats['charged_bytes']
                store._temporary = Broken()
                with self.assertRaises(MemoryError) as found:
                    store.reserve_temporary(1000)
                self.assertIs(found.exception,primary)
                self.assertEqual(store.stats['charged_bytes'],before)
                self.assertEqual(store.stats['temporary_leases'],0)
                self.assertFalse(store.stats['poisoned'])
                store.close()

    def test_unexpected_planning_failure_poisons_every_borrow_boundary(self):
        for operation in ('match','pin_range','identity'):
            with self.subTest(operation=operation):
                primary = MemoryError('rows already borrowed')
                store = PrefixBlocks(100_000)
                tokens = (1,2,3,4)
                self.publish(store,tokens,groups=('full','hidden'))
                original = store._coverage
                def fail(*args,**kwargs):
                    rows = original(*args,**kwargs)
                    self.assertTrue(rows)
                    raise primary
                with patch.object(store,'_coverage',side_effect=fail):
                    with self.assertRaises(MemoryError) as found:
                        if operation == 'match':
                            store.match(tokens,{'full':None})
                        elif operation == 'pin_range':
                            store.pin_range(tokens,'full',0,4)
                        else:
                            store.identity(tokens,{'full':None})
                self.assertIs(found.exception,primary)
                self.assertTrue(store.stats['poisoned'])
                before = store.stats['payload_bytes']
                with self.assertRaises(RuntimeError) as refused:
                    store.reserve_temporary(store.budget_bytes)
                self.assertIs(refused.exception.__cause__,primary)
                self.assertEqual(store.stats['payload_bytes'],before)
                store.close()

    def test_stats_failure_retains_failure_not_refundable_payload_borrow(self):
        primary = MemoryError('stats snapshot')
        store = PrefixBlocks(100_000)
        self.publish(store,(1,2),groups=('full',))
        with patch.object(store,'_used',side_effect=primary):
            with self.assertRaises(MemoryError) as found:
                _ = store.stats
        self.assertIs(found.exception,primary)
        self.assertTrue(store.stats['poisoned'])
        with self.assertRaises(RuntimeError) as refused:
            store.reserve_temporary(1)
        self.assertIs(refused.exception.__cause__,primary)
        store.close()

    def test_close_journal_oom_never_refunds_or_unpins_and_is_not_retried(self):
        for kind in ('read','writer','temporary'):
            with self.subTest(kind=kind):
                primary = MemoryError(kind+' remove')
                class Broken(list):
                    def remove(self,value):
                        raise primary
                store = PrefixBlocks(100_000)
                writer,_,_ = self.publish(store,(1,2),groups=('full',),finish=kind!='writer')
                if kind == 'read':
                    owner = store.match((1,2),{'full':None},require_end_group=None)
                    store._readers = Broken(store._readers)
                elif kind == 'writer':
                    owner = writer
                    store._writers = Broken(store._writers)
                else:
                    owner = store.reserve_temporary(1000)
                    store._temporary = Broken(store._temporary)
                before = store.stats
                with self.assertRaises(MemoryError) as found:
                    owner.close()
                self.assertIs(found.exception,primary)
                after = store.stats
                self.assertEqual(after['pins'],before['pins'])
                self.assertEqual(after['charged_bytes'],before['charged_bytes'])
                self.assertTrue(after['poisoned'])
                self.assertFalse(owner._closed)
                with self.assertRaises(RuntimeError) as retry:
                    owner.close()
                self.assertIs(retry.exception.__cause__,primary)
                with self.assertRaises(RuntimeError):
                    store.reserve_temporary(store.budget_bytes)

    def test_partial_span_construction_failure_is_sticky_after_read_retirement(self):
        primary = MemoryError('partial span')
        store = PrefixBlocks(100_000)
        self.publish(store,(1,2),groups=('full',))
        reader = store.match((1,2),{'full':None},require_end_group=None)
        original = reader._coverage['full']
        class BrokenRows:
            def __len__(self):
                return len(original)
            def __getitem__(self,index):
                return original[index]
            def __iter__(self):
                yield from original
                raise primary
        reader._coverage['full'] = BrokenRows()
        with self.assertRaises(MemoryError) as found:
            reader.spans('full',0,2)
        self.assertIs(found.exception,primary)
        reader.close()  # Positive consumer retirement, not cache recovery.
        self.assertTrue(store.stats['poisoned'])
        with self.assertRaises(RuntimeError):
            store.reserve_temporary(1)
        store.close()


if __name__ == '__main__':
    unittest.main()
