import React, { useState } from 'react';
import './RoleModal.css';

function RoleModal({ isOpen, onClose, onSelectRole, currentRole }) {
  const [selectedRole, setSelectedRole] = useState(currentRole || 'researcher');

  const roles = [
    {
      id: 'researcher',
      name: 'Researcher',
      title: 'Researcher/University/Institute',
      description: 'Simple interface focused on finding influential papers and authors',
      features: [
        'Search papers by authors or DOI/titles',
        'View PageRank results',
        'Citation network graph visualization',
        'Basic statistics (total papers, citations)',
        'Default parameters (α=0.85, iterations=100)'
      ],
      color: '#0047AB'
    },
    {
      id: 'data_scientist',
      name: 'Data Scientist',
      title: 'Data Scientist',
      description: 'Advanced interface with full network analysis and algorithm comparison',
      features: [
        'All Researcher features',
        '3 Algorithms: PageRank, Weighted PageRank, HITS',
        'Algorithm comparison mode (side-by-side)',
        'Customize damping factor & max iterations',
        'Network metrics panel (density, clustering, etc.)',
        'Hub & Authority scores analysis',
        'Export data (JSON/CSV)',
        'Performance metrics & correlation analysis'
      ],
      color: '#FF6B35'
    }
  ];

  const handleConfirm = () => {
    onSelectRole(selectedRole);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="role-modal-overlay">
      <div className="role-modal">
        <div className="role-modal-header">
          <h2>Welcome to Citation Network PageRank System</h2>
          <p>Please select your role to customize the interface for your needs</p>
        </div>

        <div className="role-modal-content">
          {roles.map(role => (
            <div 
              key={role.id}
              className={`role-option ${selectedRole === role.id ? 'selected' : ''}`}
              onClick={() => setSelectedRole(role.id)}
            >
              <div className="role-option-header">
                <input
                  type="radio"
                  name="role"
                  value={role.id}
                  checked={selectedRole === role.id}
                  onChange={() => setSelectedRole(role.id)}
                  className="role-radio"
                />
                <div>
                  <h3>{role.name}</h3>
                  <p className="role-subtitle">{role.title}</p>
                </div>
              </div>

              <p className="role-desc">{role.description}</p>

              <div className="role-feature-list">
                {role.features.map((feature, idx) => (
                  <span key={idx} className="feature-badge">✓ {feature}</span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="role-modal-footer">
          <button 
            className="btn-confirm" 
            onClick={handleConfirm}
          >
            Continue as {selectedRole === 'researcher' ? 'Researcher' : 'Data Scientist'} →
          </button>
        </div>
      </div>
    </div>
  );
}

export default RoleModal;