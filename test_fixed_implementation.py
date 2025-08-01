#!/usr/bin/env python3
"""
Test the fixed DQN model implementation.
"""

import torch
from ScotlandYard.core.game import TransportType
from training.deep_q.dqn_model import create_dqn_model

def test_fixed_implementation():
    """Test that the fixed implementation works correctly."""
    print("🧪 Testing Fixed DQN Implementation")
    print("=" * 50)
    
    # Test Mr. X model
    print("\n🎭 Testing Mr. X Model:")
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
    print(f"   Action size: {mrx_model.action_size}")
    
    # Test action encoding
    state = torch.randn(50)
    
    # Test single action encoding
    action_no_double = mrx_model.encode_action(42, TransportType.TAXI, use_double_move=False)
    action_with_double = mrx_model.encode_action(42, TransportType.TAXI, use_double_move=True)
    print(f"   No double move encoding: {action_no_double} (size: {action_no_double.shape[0]})")
    print(f"   With double move encoding: {action_with_double} (size: {action_with_double.shape[0]})")
    
    # Test batch query
    actions_batch_mrx = [
        (42, TransportType.TAXI, False),
        (35, TransportType.BUS, True),
        (67, TransportType.UNDERGROUND, False)
    ]
    
    q_values_mrx = mrx_model.query_batch_actions(state.unsqueeze(0).repeat(3, 1), actions_batch_mrx)
    print(f"   Batch query result: {q_values_mrx.shape} -> {q_values_mrx}")
    
    # Test detective model
    print("\n🕵️ Testing Detective Model:")
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
    print(f"   Action size: {det_model.action_size}")
    
    # Test action encoding
    action_det = det_model.encode_action(42, TransportType.TAXI, use_double_move=False)
    print(f"   Detective encoding: {action_det} (size: {action_det.shape[0]})")
    
    # Test batch query
    actions_batch_det = [
        (42, TransportType.TAXI),
        (35, TransportType.BUS),
        (67, TransportType.UNDERGROUND)
    ]
    
    q_values_det = det_model.query_batch_actions(state.unsqueeze(0).repeat(3, 1), actions_batch_det)
    print(f"   Batch query result: {q_values_det.shape} -> {q_values_det}")
    
    # Test query_batch_max_q_values
    print("\n📊 Testing Max Q-Value Queries:")
    
    valid_moves = {(42, TransportType.TAXI), (35, TransportType.BUS)}
    states_batch = state.unsqueeze(0).repeat(2, 1)
    valid_moves_batch = [valid_moves, valid_moves]
    
    max_q_mrx = mrx_model.query_batch_max_q_values(states_batch, valid_moves_batch)
    print(f"   Mr. X max Q-values: {max_q_mrx}")
    
    max_q_det = det_model.query_batch_max_q_values(states_batch, valid_moves_batch)
    print(f"   Detective max Q-values: {max_q_det}")
    
    # Test action selection
    print("\n⚡ Testing Action Selection:")
    
    mrx_action = mrx_model.select_action(state, valid_moves, epsilon=0.0, can_use_double_move=True)
    print(f"   Mr. X action (can double): {mrx_action} (length: {len(mrx_action)})")
    
    mrx_action_no_double = mrx_model.select_action(state, valid_moves, epsilon=0.0, can_use_double_move=False)
    print(f"   Mr. X action (no double): {mrx_action_no_double} (length: {len(mrx_action_no_double)})")
    
    det_action = det_model.select_action(state, valid_moves, epsilon=0.0, can_use_double_move=False)
    print(f"   Detective action: {det_action} (length: {len(det_action)})")
    
    print("\n✅ Verification:")
    assert mrx_model.action_size == 3, f"Mr. X action_size should be 3, got {mrx_model.action_size}"
    assert det_model.action_size == 2, f"Detective action_size should be 2, got {det_model.action_size}"
    assert len(mrx_action) == 3, f"Mr. X action should be 3-tuple, got {len(mrx_action)}"
    assert len(mrx_action_no_double) == 3, f"Mr. X action (no double) should be 3-tuple, got {len(mrx_action_no_double)}"
    assert len(det_action) == 2, f"Detective action should be 2-tuple, got {len(det_action)}"
    assert mrx_action_no_double[2] == False, f"Mr. X no double action should have use_double_move=False"
    
    print("   ✓ All action sizes are correct!")
    print("   ✓ All action tuple lengths are correct!")
    print("   ✓ Batch queries work properly!")
    
    print("\n🎉 All tests passed! The implementation is fixed!")

if __name__ == "__main__":
    test_fixed_implementation()
