#!/usr/bin/env python3
"""
Test script for the gym-zelda-pygame environment.
This script demonstrates how to use the environment and tests basic functionality.
"""

import gymnasium as gym
import gym_zelda_pygame
import numpy as np
import pygame

def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing gym-zelda-pygame environment...")
    
    # Create environment
    try:
        env = gym.make("ZeldaCC-v0", render_mode="rgb_array")
        print("Environment created successfully")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return False
    
    # Test reset
    try:
        obs, info = env.reset()
        print(f"Environment reset. Observation shape: {obs.shape}")
        print(f"  Info: {info}")
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        env.close()
        return False
    
    # Test action space
    print(f"✓ Action space: {env.action_space}")
    print(f"✓ Observation space: {env.observation_space}")
    
    # Test a few random steps
    try:
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.4f}, done={done}, truncated={truncated}")
            
            if done or truncated:
                print("  Episode ended, resetting...")
                obs, info = env.reset()
                break
        
        print("Steps executed successfully")
    except Exception as e:
        print(f"Failed during step execution: {e}")
        env.close()
        return False
    
    # Test render
    try:
        frame = env.render()
        if frame is not None:
            print(f"Render successful. Frame shape: {frame.shape}")
        else:
            print("Render successful (no return value)")
    except Exception as e:
        print(f"Failed to render: {e}")
    
    # Clean up
    env.close()
    print("Environment closed successfully")
    return True

def test_with_manual_actions():
    """Test environment with specific manual actions."""
    print("\nTesting with manual actions...")
    
    try:
        env = gym.make("ZeldaCC-v0", render_mode="rgb_array")
        obs, info = env.reset()
        
        # Test movement actions
        actions_to_test = [
            (1, "up"),
            (2, "down"), 
            (3, "left"),
            (4, "right"),
            (5, "attack"),
            (6, "magic"),
            (0, "no action")
        ]
        
        print("Testing specific actions:")
        for action_id, action_name in actions_to_test:
            obs, reward, done, truncated, info = env.step(action_id)
            print(f"  {action_name}: reward={reward:.4f}, done={done}")
            
            if done or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✓ Manual actions test completed")
        return True
        
    except Exception as e:
        print(f"✗ Manual actions test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== gym-zelda-pygame Environment Test ===\n")
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    
    # Test manual actions if basic test passed
    if basic_test_passed:
        manual_test_passed = test_with_manual_actions()
    else:
        manual_test_passed = False
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Basic functionality: {'PASS' if basic_test_passed else 'FAIL'}")
    print(f"Manual actions: {'PASS' if manual_test_passed else 'FAIL'}")
    
    if basic_test_passed and manual_test_passed:
        print("\nAll tests passed! The environment is working correctly.")
        print("\nYou can now use the environment in your RL training scripts:")
        print("```python")
        print("import gymnasium as gym")
        print("import gym_zelda_pygame")
        print()
        print("env = gym.make('ZeldaCC-v0', render_mode='human')")
        print("obs, info = env.reset()")
        print()
        print("for _ in range(1000):")
        print("    action = env.action_space.sample()")
        print("    obs, reward, done, truncated, info = env.step(action)")
        print("    if done or truncated:")
        print("        obs, info = env.reset()")
        print("```")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()