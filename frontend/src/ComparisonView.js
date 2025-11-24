import React, { useState } from 'react';
import './ComparisonView.css';
import { Scatter } from 'react-chartjs-2';

function ComparisonView({ comparisonData, algorithms }) {
  const [activeTab, setActiveTab] = useState('side-by-side');

  if (!comparisonData || !comparisonData.algorithms) {
    return null;
  }

  const algorithmInfo = {
    pagerank: { name: 'PageRank', color: '#0047AB', scoreKey: 'pagerank' },
    weighted_pagerank: { name: 'Weighted PageRank', color: '#FF6B35', scoreKey: 'weighted_pagerank' },
    hits: { name: 'HITS Authority', color: '#4CAF50', scoreKey: 'authority_score' }
  };

  // Prepare data for side-by-side comparison
  const prepareComparisonTable = () => {
    const maxRows = 20;
    const rows = [];

    for (let i = 0; i < maxRows; i++) {
      const row = { rank: i + 1 };

      algorithms.forEach(algoId => {
        const algoData = comparisonData.algorithms[algoId];
        if (!algoData) return;

        let result;
        if (algoId === 'hits') {
          result = algoData.authority_results?.[i];
        } else {
          result = algoData.results?.[i];
        }

        if (result) {
          row[algoId] = {
            title: result.title,
            score: result[algorithmInfo[algoId]?.scoreKey] || 0,
            year: result.year,
            citations: result.citationCount
          };
        }
      });

      rows.push(row);
    }

    return rows;
  };

  // Prepare correlation scatter plot
  const prepareScatterData = () => {
    if (algorithms.length < 2) return null;

    const algo1 = algorithms[0];
    const algo2 = algorithms[1];

    const data1 = comparisonData.algorithms[algo1];
    const data2 = comparisonData.algorithms[algo2];

    if (!data1 || !data2) return null;

    const results1 = algo1 === 'hits' ? data1.authority_results : data1.results;
    const results2 = algo2 === 'hits' ? data2.authority_results : data2.results;

    const scoreKey1 = algorithmInfo[algo1].scoreKey;
    const scoreKey2 = algorithmInfo[algo2].scoreKey;

    // Match papers by ID
    const paperMap1 = {};
    results1.forEach(r => {
      paperMap1[r.paperId] = r[scoreKey1];
    });

    const scatterPoints = [];
    results2.forEach(r => {
      if (paperMap1[r.paperId] !== undefined) {
        scatterPoints.push({
          x: paperMap1[r.paperId],
          y: r[scoreKey2],
          title: r.title.substring(0, 30)
        });
      }
    });

    return {
      datasets: [{
        label: `${algorithmInfo[algo1].name} vs ${algorithmInfo[algo2].name}`,
        data: scatterPoints,
        backgroundColor: algorithmInfo[algo1].color,
        borderColor: algorithmInfo[algo2].color,
        borderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 8
      }]
    };
  };

  const scatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const point = context.raw;
            return [
              `${point.title || 'Paper'}`,
              `${algorithmInfo[algorithms[0]].name}: ${point.x.toFixed(6)}`,
              `${algorithmInfo[algorithms[1]].name}: ${point.y.toFixed(6)}`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: algorithmInfo[algorithms[0]]?.name + ' Score'
        }
      },
      y: {
        title: {
          display: true,
          text: algorithmInfo[algorithms[1]]?.name + ' Score'
        }
      }
    }
  };

  const comparisonTable = prepareComparisonTable();
  const scatterData = prepareScatterData();

  return (
    <div className="comparison-view-container">
      <div className="comparison-header">
        <h2>Algorithm Comparison Results</h2>
        <div className="comparison-tabs">
          <button
            className={`tab-btn ${activeTab === 'side-by-side' ? 'active' : ''}`}
            onClick={() => setActiveTab('side-by-side')}
          >
            Side-by-Side
          </button>
          <button
            className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            Performance Metrics
          </button>
          {algorithms.length >= 2 && (
            <button
              className={`tab-btn ${activeTab === 'correlation' ? 'active' : ''}`}
              onClick={() => setActiveTab('correlation')}
            >
              Correlation Analysis
            </button>
          )}
        </div>
      </div>

      {/* Side-by-Side Table */}
      {activeTab === 'side-by-side' && (
        <div className="comparison-table-container">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Rank</th>
                {algorithms.map(algoId => {
                  const info = algorithmInfo[algoId];
                  return (
                    <th key={algoId} style={{ borderBottom: `4px solid ${info.color}` }}>
                      <div className="algo-header-cell">
                        <span className="algo-icon">{info.icon}</span>
                        <span>{info.name}</span>
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {comparisonTable.map((row, idx) => (
                <tr key={idx} className={idx < 3 ? 'top-rank' : ''}>
                  <td className="rank-cell">
                    <strong>#{row.rank}</strong>
                    {idx === 0 && '🥇'}
                    {idx === 1 && '🥈'}
                    {idx === 2 && '🥉'}
                  </td>
                  {algorithms.map(algoId => {
                    const paper = row[algoId];
                    return (
                      <td key={algoId} className="paper-cell">
                        {paper ? (
                          <>
                            <div className="paper-title">{paper.title}</div>
                            <div className="paper-meta">
                              <span className="paper-score" style={{ color: algorithmInfo[algoId].color }}>
                                Score: {paper.score.toFixed(6)}
                              </span>
                              <span className="paper-year">Year: {paper.year || 'N/A'}</span>
                              <span className="paper-citations">Citations: {paper.citations}</span>
                            </div>
                          </>
                        ) : (
                          <span className="no-data">-</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Performance Metrics */}
      {activeTab === 'metrics' && (
        <>
          {/* FIX: Separated metrics grid from statistical section */}
          <div className="metrics-grid">
            {algorithms.map(algoId => {
              const algoData = comparisonData.algorithms[algoId];
              const info = algorithmInfo[algoId];
              
              return (
                <div key={algoId} className="metric-card" style={{ borderTop: `6px solid ${info.color}` }}>
                  <div className="metric-card-header">
                    <span className="algo-icon">{info.icon}</span>
                    <h3>{info.name}</h3>
                  </div>
                  
                  <div className="metric-content">
                    <div className="metric-row">
                      <span className="metric-label">⏱️ Computation Time:</span>
                      <span className="metric-value">{algoData.performance.computation_time}s</span>
                    </div>
                    
                    <div className="metric-row">
                      <span className="metric-label">🔄 Iterations:</span>
                      <span className="metric-value">{algoData.performance.iterations}</span>
                    </div>
                    
                    <div className="metric-row">
                      <span className="metric-label">📄 Papers Analyzed:</span>
                      <span className="metric-value">{algoData.performance.papers_analyzed}</span>
                    </div>

                    {algoId === 'hits' && (
                      <div className="metric-note">
                        <small>💡 HITS produces both Authority and Hub scores</small>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* FIX: Statistical Comparisons - Now completely separate */}
          {comparisonData.correlations && Object.keys(comparisonData.correlations).length > 0 && (
            <div className="correlation-metrics">
              <h3>Statistical Comparisons</h3>
              
              {Object.entries(comparisonData.correlations).map(([key, value]) => (
                <div key={key} className="correlation-row">
                  <span className="correlation-label">
                    {key.replace(/_/g, ' ').replace(/vs/g, '⚡')}:
                  </span>
                  <div className="correlation-bar-container">
                    <div 
                      className="correlation-bar" 
                      style={{ 
                        width: `${Math.abs(value) * 100}%`,
                        background: value > 0.7 ? '#4CAF50' : value > 0.4 ? '#FF9800' : '#f44336'
                      }}
                    >
                      <span className="correlation-value">{value.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              ))}

              {comparisonData.overlaps && Object.entries(comparisonData.overlaps).map(([key, value]) => (
                <div key={key} className="overlap-row">
                  <span className="overlap-label">
                    {key.replace(/_/g, ' ').replace(/top10/g, 'Top-10 Overlap')}:
                  </span>
                  <span className="overlap-value">{(value * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Correlation Scatter Plot */}
      {activeTab === 'correlation' && scatterData && (
        <div className="correlation-chart-container">
          <h3>Score Correlation: {algorithmInfo[algorithms[0]].name} vs {algorithmInfo[algorithms[1]].name}</h3>
          <div style={{ height: '500px' }}>
            <Scatter data={scatterData} options={scatterOptions} />
          </div>
          <div className="correlation-interpretation">
            <h4>Interpretation:</h4>
            <ul>
              <li>Points closer to diagonal line = Higher agreement between algorithms</li>
              <li>Scattered points = Different ranking perspectives</li>
              <li>Correlation coefficient: {comparisonData.correlations?.[`${algorithms[0]}_vs_${algorithms[1].split('_')[0]}`]?.toFixed(3) || 'N/A'}</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default ComparisonView;