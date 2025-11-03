import React from 'react';
import { Bar, Doughnut, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js';
import './NetworkMetrics.css';

ChartJS.register(
  ArcElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const NetworkMetrics = ({ metrics, results }) => {
  if (!metrics) return null;

  // Prepare data for In-Degree Distribution chart
  const inDegreeData = {
    labels: results.map((r, i) => `#${i + 1}`),
    datasets: [{
      label: 'In-Degree (Citations)',
      data: metrics.in_degree,
      backgroundColor: 'rgba(54, 162, 235, 0.8)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 2,
      borderRadius: 8
    }]
  };

  // Prepare data for Out-Degree Distribution chart
  const outDegreeData = {
    labels: results.map((r, i) => `#${i + 1}`),
    datasets: [{
      label: 'Out-Degree (References)',
      data: metrics.out_degree,
      backgroundColor: 'rgba(255, 99, 132, 0.8)',
      borderColor: 'rgba(255, 99, 132, 1)',
      borderWidth: 2,
      borderRadius: 8
    }]
  };

  // Prepare data for Node Types Doughnut
  const nodeTypesData = {
    labels: ['Strongly Connected', 'Dangling Nodes', 'Isolated Nodes', 'Others'],
    datasets: [{
      data: [
        metrics.strongly_connected_nodes,
        metrics.dangling_nodes,
        metrics.isolated_nodes,
        Math.max(0, metrics.total_nodes - metrics.strongly_connected_nodes - metrics.dangling_nodes - metrics.isolated_nodes)
      ],
      backgroundColor: [
        'rgba(75, 192, 192, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(201, 203, 207, 0.8)'
      ],
      borderWidth: 2,
      borderColor: '#fff'
    }]
  };

  // Prepare Radar chart for top 5 nodes
  const topNodes = results.slice(0, 5);
  const radarData = {
    labels: ['PageRank', 'Hub Score', 'Authority Score', 'In-Degree', 'Out-Degree'],
    datasets: topNodes.map((result, idx) => {
      const colors = [
        'rgba(255, 99, 132, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(255, 206, 86, 0.6)',
        'rgba(153, 102, 255, 0.6)'
      ];
      
      // UPDATED: Use title instead of url
      const label = result.title && result.title.length > 30 
        ? result.title.substring(0, 30) + '...' 
        : result.title || 'Unknown';
      
      return {
        label: label,
        data: [
          result.pagerank * 10, // UPDATED: pagerank instead of rank
          metrics.hub_scores[idx] * 10,
          metrics.authority_scores[idx] * 10,
          metrics.in_degree[idx],
          metrics.out_degree[idx]
        ],
        backgroundColor: colors[idx],
        borderColor: colors[idx].replace('0.6', '1'),
        borderWidth: 2
      };
    })
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, position: 'top' },
      tooltip: { enabled: true }
    },
    scales: {
      y: { beginAtZero: true }
    }
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, position: 'bottom' }
    },
    scales: {
      r: {
        beginAtZero: true,
        ticks: { stepSize: 2 }
      }
    }
  };

  return (
    <div className="network-metrics-container">
      <div className="metrics-header">
        <h3>🔢 Citation Network Metrics Panel</h3>
        <p className="metrics-description">
          💡 Comprehensive analysis of citation network structure, connectivity, and paper importance
        </p>
      </div>

      {/* Overview Cards */}
      <div className="metrics-overview">
        <div className="metric-card">
          <div className="metric-icon">📄</div>
          <div className="metric-content">
            <div className="metric-value">{metrics.total_nodes}</div>
            <div className="metric-label">Total Papers</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🔗</div>
          <div className="metric-content">
            <div className="metric-value">{metrics.total_edges}</div>
            <div className="metric-label">Total Citations</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📈</div>
          <div className="metric-content">
            <div className="metric-value">{(metrics.density * 100).toFixed(1)}%</div>
            <div className="metric-label">Network Density</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🎯</div>
          <div className="metric-content">
            <div className="metric-value">{metrics.avg_clustering_coefficient.toFixed(3)}</div>
            <div className="metric-label">Avg Clustering</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📥</div>
          <div className="metric-content">
            <div className="metric-value">{metrics.avg_in_degree.toFixed(1)}</div>
            <div className="metric-label">Avg Citations</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📤</div>
          <div className="metric-content">
            <div className="metric-value">{metrics.avg_out_degree.toFixed(1)}</div>
            <div className="metric-label">Avg References</div>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="metrics-charts-grid">
        <div className="chart-box">
          <h4>In-Degree Distribution (Citations Received)</h4>
          <p className="chart-description">Number of times each paper is cited</p>
          <div style={{ height: '300px' }}>
            <Bar data={inDegreeData} options={chartOptions} />
          </div>
        </div>

        <div className="chart-box">
          <h4>Out-Degree Distribution (References Made)</h4>
          <p className="chart-description">Number of papers each paper references</p>
          <div style={{ height: '300px' }}>
            <Bar data={outDegreeData} options={chartOptions} />
          </div>
        </div>

        <div className="chart-box">
          <h4>Paper Types Distribution</h4>
          <p className="chart-description">Classification of papers by connectivity</p>
          <div style={{ height: '300px' }}>
            <Doughnut data={nodeTypesData} options={{ responsive: true, maintainAspectRatio: false }} />
          </div>
        </div>

        <div className="chart-box">
          <h4>Top 5 Papers - Multi-Metric Analysis</h4>
          <p className="chart-description">Comprehensive comparison of top-ranked papers</p>
          <div style={{ height: '300px' }}>
            <Radar data={radarData} options={radarOptions} />
          </div>
        </div>
      </div>

      {/* Hubs and Authorities */}
      <div className="hub-authority-section">
        <div className="hub-box">
          <h4>🌟 Top Hubs (High References)</h4>
          <p className="section-description">Papers that reference many other papers</p>
          {metrics.hubs && metrics.hubs.length > 0 ? (
            <ul className="hub-list">
              {metrics.hubs.map((hub, idx) => (
                <li key={idx}>
                  <span className="rank-badge">#{idx + 1}</span>
                  <a href={hub.url || '#'} target="_blank" rel="noopener noreferrer" className="hub-url">
                    {hub.url || 'Paper ' + (idx + 1)}
                  </a>
                  <span className="hub-score">
                    📤 {hub.out_degree} refs | Score: {(hub.score * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-data">No significant hubs detected</p>
          )}
        </div>

        <div className="authority-box">
          <h4>👑 Top Authorities (Highly Cited)</h4>
          <p className="section-description">Papers that receive many citations</p>
          {metrics.authorities && metrics.authorities.length > 0 ? (
            <ul className="authority-list">
              {metrics.authorities.map((auth, idx) => (
                <li key={idx}>
                  <span className="rank-badge gold">#{idx + 1}</span>
                  <a href={auth.url || '#'} target="_blank" rel="noopener noreferrer" className="authority-url">
                    {auth.url || 'Paper ' + (idx + 1)}
                  </a>
                  <span className="authority-score">
                    📥 {auth.in_degree} cites | Score: {(auth.score * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-data">No significant authorities detected</p>
          )}
        </div>
      </div>

      {/* Insights Panel */}
      <div className="insights-panel">
        <h4>💡 Citation Network Insights</h4>
        <div className="insights-grid">
          <div className="insight-item">
            <span className="insight-icon">
              {metrics.density > 0.5 ? '✅' : metrics.density > 0.2 ? '⚠️' : '❌'}
            </span>
            <div>
              <strong>Density: {(metrics.density * 100).toFixed(1)}%</strong>
              <p>
                {metrics.density > 0.5
                  ? 'Highly connected research network with strong cross-citations'
                  : metrics.density > 0.2
                  ? 'Moderately connected research network'
                  : 'Sparse network with limited cross-citations'}
              </p>
            </div>
          </div>

          <div className="insight-item">
            <span className="insight-icon">
              {metrics.dangling_nodes === 0 ? '✅' : '⚠️'}
            </span>
            <div>
              <strong>{metrics.dangling_nodes} Paper(s) With No References</strong>
              <p>
                {metrics.dangling_nodes === 0
                  ? 'All papers have outbound references - comprehensive literature review'
                  : `${metrics.dangling_nodes} paper(s) have no outbound references`}
              </p>
            </div>
          </div>

          <div className="insight-item">
            <span className="insight-icon">
              {metrics.isolated_nodes === 0 ? '✅' : '❌'}
            </span>
            <div>
              <strong>{metrics.isolated_nodes} Isolated Paper(s)</strong>
              <p>
                {metrics.isolated_nodes === 0
                  ? 'No isolated papers - fully connected citation network'
                  : `${metrics.isolated_nodes} paper(s) are completely isolated`}
              </p>
            </div>
          </div>

          <div className="insight-item">
            <span className="insight-icon">
              {metrics.avg_clustering_coefficient > 0.6 ? '✅' : metrics.avg_clustering_coefficient > 0.3 ? '⚠️' : '❌'}
            </span>
            <div>
              <strong>Clustering: {metrics.avg_clustering_coefficient.toFixed(3)}</strong>
              <p>
                {metrics.avg_clustering_coefficient > 0.6
                  ? 'High clustering - papers form tight research communities'
                  : metrics.avg_clustering_coefficient > 0.3
                  ? 'Moderate clustering - some research community structure'
                  : 'Low clustering - papers are loosely connected'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkMetrics;