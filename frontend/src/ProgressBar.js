import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ progress, message, stage }) => {
  // Map stage to emoji and color
  const stageConfig = {
    init: { emoji: '', color: '#3498db' },
    authors: { emoji: '👤', color: '#9b59b6' },
    citations: { emoji: '🔗', color: '#e74c3c' },
    calculation: { emoji: '🧮', color: '#f39c12' },
    finalize: { emoji: '📊', color: '#27ae60' },
  };

  const config = stageConfig[stage] || stageConfig.init;

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-header">
        <span className="progress-emoji">{config.emoji}</span>
        <span className="progress-message">{message}</span>
      </div>
      
      <div className="progress-bar-wrapper">
        <div  
          className="progress-bar-fill" 
          style={{ 
            width: `${progress}%`,
            backgroundColor: config.color,
            transition: 'width 0.3s ease-in-out'
          }}
        >
          <span className="progress-bar-text">{progress}%</span>
        </div>
      </div>
      
      <div className="progress-bar-stages">
        <div className={`stage ${progress >= 5 ? 'active' : ''} ${progress > 35 ? 'complete' : ''}`}>
          <span>👤 Authors</span>
        </div>
        <div className={`stage ${progress >= 35 ? 'active' : ''} ${progress > 85 ? 'complete' : ''}`}>
          <span>🔗 Citations</span>
        </div>
        <div className={`stage ${progress >= 85 ? 'active' : ''} ${progress > 95 ? 'complete' : ''}`}>
          <span>🧮 Calculate</span>
        </div>
        <div className={`stage ${progress >= 95 ? 'active' : ''} ${progress === 100 ? 'complete' : ''}`}>
          <span>📊 Results</span>
        </div>
      </div>
    </div>
  );
};

export default ProgressBar;