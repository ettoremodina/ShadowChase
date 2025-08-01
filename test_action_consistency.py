#!/usr/bin/env python3
"""
Test to verify that action tuple lengths are consistent for Mr. X.
"""

import torch
from ScotlandYard.core.game import TransportType
from training.deep_q.dqn_model import create_dqn_model

def test_mrx_action_consistency():
    """Test that Mr. X always returns 3-tuples regardless of double move availability."""
    print("🧪 Testing Mr. X Action Tuple Consistency")
    print("=" * 50)
    
    # Create Mr. X model (action_size=3)
    mrx_config = {
        'network_parameters': {
            'action_size': 3,
            'hidden_layers': [64, 32],
            'dropout_rate': 0.1
        },
        'feature_extraction': {
            'input_size': 50
        }
    }
    
    mrx_model = create_dqn_model(mrx_config)
    mrx_model.eval()
    
    # Create detective model (action_size=2)
    det_config = {
        'network_parameters': {
            'action_size': 2,
            'hidden_layers': [64, 32],
            'dropout_rate': 0.1
        },
        'feature_extraction': {
            'input_size': 50
        }
    }
    
    det_model = create_dqn_model(det_config)
    det_model.eval()
    
    # Test data
    state = torch.randn(50)
    valid_moves = {(42, TransportType.TAXI), (35, TransportType.BUS)}
    
    print("\n🎭 Testing Mr. X Actions:")
    
    # Test 1: Mr. X with double move available
    mrx_action_with_double = mrx_model.select_action(
        state, valid_moves, epsilon=0.0, can_use_double_move=True
    )
    print(f"   With double move available: {mrx_action_with_double}")
    print(f"   Length: {len(mrx_action_with_double)} (should be 3)")
    
    # Test 2: Mr. X without double move available
    mrx_action_no_double = mrx_model.select_action(
        state, valid_moves, epsilon=0.0, can_use_double_move=False
    )
    print(f"   Without double move:        {mrx_action_no_double}")
    print(f"   Length: {len(mrx_action_no_double)} (should be 3)")
    
    # Test 3: Mr. X random action with double move
    mrx_random_with_double = mrx_model.select_action(
        state, valid_moves, epsilon=1.0, can_use_double_move=True
    )
    print(f"   Random with double:         {mrx_random_with_double}")
    print(f"   Length: {len(mrx_random_with_double)} (should be 3)")
    
    # Test 4: Mr. X random action without double move
    mrx_random_no_double = mrx_model.select_action(
        state, valid_moves, epsilon=1.0, can_use_double_move=False
    )
    print(f"   Random without double:      {mrx_random_no_double}")
    print(f"   Length: {len(mrx_random_no_double)} (should be 3)")
    
    print("\n🕵️ Testing Detective Actions:")
    
    # Test 5: Detective action (should always be 2-tuple)
    det_action = det_model.select_action(
        state, valid_moves, epsilon=0.0, can_use_double_move=False
    )
    print(f"   Detective action:           {det_action}")
    print(f"   Length: {len(det_action)} (should be 2)")
    
    # Test 6: Detective random action
    det_random = det_model.select_action(
        state, valid_moves, epsilon=1.0, can_use_double_move=False
    )
    print(f"   Detective random:           {det_random}")
    print(f"   Length: {len(det_random)} (should be 2)")
    
    print("\n📊 Testing query_multiple_actions:")
    
    # Test Mr. X query with double move
    q_vals_mrx_double, actions_mrx_double = mrx_model.query_multiple_actions(
        state, valid_moves, can_use_double_move=True
    )
    print(f"   Mr. X with double move: {len(actions_mrx_double)} actions")
    for i, action in enumerate(actions_mrx_double):
        print(f"     {i+1}. {action} (length: {len(action)})")
    
    # Test Mr. X query without double move
    q_vals_mrx_no_double, actions_mrx_no_double = mrx_model.query_multiple_actions(
        state, valid_moves, can_use_double_move=False
    )
    print(f"   Mr. X without double move: {len(actions_mrx_no_double)} actions")
    for i, action in enumerate(actions_mrx_no_double):
        print(f"     {i+1}. {action} (length: {len(action)})")
    
    # Test Detective query
    q_vals_det, actions_det = det_model.query_multiple_actions(
        state, valid_moves, can_use_double_move=False
    )
    print(f"   Detective: {len(actions_det)} actions")
    for i, action in enumerate(actions_det):
        print(f"     {i+1}. {action} (length: {len(action)})")
    
    print("\n✅ Verification:")
    
    # Verify Mr. X always returns 3-tuples
    assert len(mrx_action_with_double) == 3, f"Mr. X with double should return 3-tuple, got {len(mrx_action_with_double)}"
    assert len(mrx_action_no_double) == 3, f"Mr. X without double should return 3-tuple, got {len(mrx_action_no_double)}"
    assert len(mrx_random_with_double) == 3, f"Mr. X random with double should return 3-tuple, got {len(mrx_random_with_double)}"
    assert len(mrx_random_no_double) == 3, f"Mr. X random without double should return 3-tuple, got {len(mrx_random_no_double)}"
    
    # Verify detectives always return 2-tuples
    assert len(det_action) == 2, f"Detective should return 2-tuple, got {len(det_action)}"
    assert len(det_random) == 2, f"Detective random should return 2-tuple, got {len(det_random)}"
    
    # Verify query_multiple_actions consistency
    for action in actions_mrx_double:
        assert len(action) == 3, f"Mr. X query action should be 3-tuple, got {len(action)}"
    for action in actions_mrx_no_double:
        assert len(action) == 3, f"Mr. X query action should be 3-tuple, got {len(action)}"
    for action in actions_det:
        assert len(action) == 2, f"Detective query action should be 2-tuple, got {len(action)}"
    
    # Verify Mr. X without double move has use_double_move=False
    assert mrx_action_no_double[2] == False, f"Mr. X without double move should have use_double_move=False"
    
    print("   ✓ All Mr. X actions return 3-tuples!")
    print("   ✓ All detective actions return 2-tuples!")
    print("   ✓ Mr. X without double move correctly sets use_double_move=False!")
    
    print("\n🎉 All consistency tests passed!")

if __name__ == "__main__":
    test_mrx_action_consistency()
