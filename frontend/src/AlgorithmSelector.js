import React, { useState } from 'react';
import './AlgorithmSelector.css';

function AlgorithmSelector({ onAlgorithmChange, onParametersChange, selectedAlgorithms, parameters }) {
  const [compareMode, setCompareMode] = useState(false);

  const algorithms = [
    {
      id: 'pagerank',
      name: 'PageRank (Standard)',
      description: 'Google\'s original algorithm - measures global importance',
      formula: 'PR(A) = (1-d)/N + d × Σ(PR(T_i)/C(T_i))',
      parameters: ['damping_factor', 'max_iterations'],
      color: '#0047AB'
    },
    {
      id: 'weighted_pagerank',
      name: 'Weighted PageRank',
      description: 'Citation-weighted variant - higher weight for highly-cited papers',
      formula: 'PR(A) = (1-d)/N + d × Σ(PR(T_i) × W(T_i,A))',
      parameters: ['damping_factor', 'max_iterations'],
      color: '#FF6B35'
    },
    {
      id: 'hits',
      name: 'HITS Algorithm',
      description: 'Computes Hub and Authority scores separately',
      formula: 'Authority(A) = Σ Hub(T_i), Hub(A) = Σ Authority(T_i)',
      parameters: ['max_iterations'],
      color: '#4CAF50'
    }
  ];

  const handleAlgorithmToggle = (algoId) => {
    if (compareMode) {
      // Multi-select mode
      let newSelection;
      if (selectedAlgorithms.includes(algoId)) {
        newSelection = selectedAlgorithms.filter(id => id !== algoId);
      } else {
        newSelection = [...selectedAlgorithms, algoId];
      }
      onAlgorithmChange(newSelection);
    } else {
      // Single select mode
      onAlgorithmChange([algoId]);
    }
  };

  const handleCompareModeToggle = () => {
    const newMode = !compareMode;
    setCompareMode(newMode);
    
    if (!newMode && selectedAlgorithms.length > 1) {
      // Switch to single mode - keep first selection
      onAlgorithmChange([selectedAlgorithms[0]]);
    }
  };

  return (
    <div className="algorithm-selector-container">
      <div className="selector-header">
        <h3>Algorithm Selection</h3>
        <label className="compare-mode-toggle">
          <input
            type="checkbox"
            checked={compareMode}
            onChange={handleCompareModeToggle}
          />
          <span className="toggle-label">
            {compareMode ? 'Comparison Mode (Select Multiple)' : 'Single Algorithm Mode'}
          </span>
        </label>
      </div>

      <div className="algorithms-grid">
        {algorithms.map(algo => {
          const isSelected = selectedAlgorithms.includes(algo.id);
          
          return (
            <div
              key={algo.id}
              className={`algorithm-card ${isSelected ? 'selected' : ''}`}
              onClick={() => handleAlgorithmToggle(algo.id)}
              style={{
                borderColor: isSelected ? algo.color : '#e8f0fe'
              }}
            >
              <div className="algorithm-header">
                {compareMode ? (
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => {}}
                    className="algo-checkbox"
                  />
                ) : (
                  <input
                    type="radio"
                    checked={isSelected}
                    onChange={() => {}}
                    className="algo-radio"
                  />
                )}
                <span className="algo-icon">{algo.icon}</span>
                <h4 style={{ color: algo.color }}>{algo.name}</h4>
              </div>

              <p className="algo-description">{algo.description}</p>

              <div className="algo-formula">
                <strong>Formula:</strong>
                <code>{algo.formula}</code>
              </div>

              {isSelected && (
                <div className="selected-badge" style={{ background: algo.color }}>
                  ✓ Selected
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Parameters Section */}
      <div className="parameters-section">
        <h4>⚙️ Algorithm Parameters</h4>
        
        <div className="parameters-grid">
          {selectedAlgorithms.some(id => ['pagerank', 'weighted_pagerank'].includes(id)) && (
            <div className="parameter-control">
              <label>
                Damping Factor (d):
                <span className="param-value">{parameters.damping_factor}</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="0.99"
                step="0.01"
                value={parameters.damping_factor}
                onChange={(e) => onParametersChange({
                  ...parameters,
                  damping_factor: parseFloat(e.target.value)
                })}
                className="param-slider"
              />
              <small>Probability of following citations (default: 0.85)</small>
            </div>
          )}

          <div className="parameter-control">
            <label>
              Max Iterations:
              <span className="param-value">{parameters.max_iterations}</span>
            </label>
            <input
              type="range"
              min="10"
              max="500"
              step="10"
              value={parameters.max_iterations}
              onChange={(e) => onParametersChange({
                ...parameters,
                max_iterations: parseInt(e.target.value)
              })}
              className="param-slider"
            />
            <small>Maximum iterations for convergence (default: 100)</small>
          </div>
        </div>
      </div>

      {compareMode && selectedAlgorithms.length >= 2 && (
        <div className="comparison-info">
          <strong>Comparing {selectedAlgorithms.length} algorithms:</strong>
          {selectedAlgorithms.map(id => {
            const algo = algorithms.find(a => a.id === id);
            return (
              <span key={id} className="selected-algo-tag" style={{ background: algo.color }}>
                {algo.icon} {algo.name}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default AlgorithmSelector;