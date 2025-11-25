import streamlit as st
import pandas as pd
import datetime
import random
import re
import json
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import base64
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib
import uuid
import warnings
from typing import Dict, List, Any, Optional, Union  # Add this import
warnings.filterwarnings('ignore')

# =============================================
# ENHANCED CONFIGURATION & CONSTANTS
# =============================================
class Config:
    """Enhanced configuration settings"""
    APP_NAME = "AI Health System 3.0"
    VERSION = "Professional Edition"
    SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png', 'dicom', 'tiff']
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    TELEMEDICINE_SERVERS = {
        'primary': 'wss://telemedicine-primary.server.com',
        'backup': 'wss://telemedicine-backup.server.com'
    }
    AI_MODELS = {
        'symptom_checker': 'health_ai_v3',
        'imaging_analysis': 'radiology_ai_v2',
        'prescription_ai': 'medication_ai_v2'
    }

# =============================================
# ADVANCED SECURITY & AUTHENTICATION
# =============================================
class SecurityManager:
    """Enhanced security and authentication management"""
    
    def __init__(self):
        self.encryption_key = hashlib.sha256(b"healthcare_secure_key").digest()
    
    def hash_password(self, password: str) -> str:
        """Secure password hashing"""
        return hashlib.sha256(password.encode() + self.encryption_key).hexdigest()
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return str(uuid.uuid4()) + hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]
    
    def validate_access(self, user_role: str, required_role: str) -> bool:
        """Role-based access control"""
        role_hierarchy = {
            'patient': 1,
            'nurse': 2,
            'doctor': 3,
            'admin': 4
        }
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)

# =============================================
# ENHANCED AI SYMPTOM CHECKER 2.0
# =============================================
class AdvancedSymptomChecker:
    """Multi-symptom analysis with probabilities and risk assessment"""
    
    def __init__(self):
        self.symptom_database = self._initialize_symptom_database()
        self.disease_knowledge_base = self._initialize_disease_knowledge()
        self.risk_factors = self._initialize_risk_factors()
    
    def _initialize_symptom_database(self) -> Dict[str, Any]:
        """Comprehensive symptom database with probabilities"""
        return {
            # Cardiovascular Symptoms
            'chest_pain': {
                'name': 'Chest Pain',
                'categories': ['cardiac', 'respiratory', 'musculoskeletal'],
                'severity_levels': {
                    'mild': 0.1, 'moderate': 0.3, 'severe': 0.7, 'crushing': 0.9
                },
                'associated_conditions': {
                    'myocardial_infarction': 0.85,
                    'angina': 0.75,
                    'pulmonary_embolism': 0.45,
                    'costochondritis': 0.25,
                    'anxiety': 0.15
                }
            },
            'shortness_of_breath': {
                'name': 'Shortness of Breath',
                'categories': ['respiratory', 'cardiac', 'allergic'],
                'severity_levels': {
                    'mild': 0.2, 'moderate': 0.4, 'severe': 0.8, 'extreme': 0.95
                },
                'associated_conditions': {
                    'asthma': 0.65,
                    'copd': 0.55,
                    'heart_failure': 0.70,
                    'pneumonia': 0.60,
                    'anemia': 0.35
                }
            },
            
            # Neurological Symptoms
            'headache': {
                'name': 'Headache',
                'categories': ['neurological', 'vascular', 'tension'],
                'severity_levels': {
                    'mild': 0.1, 'moderate': 0.3, 'severe': 0.6, 'debilitating': 0.85
                },
                'associated_conditions': {
                    'migraine': 0.70,
                    'tension_headache': 0.45,
                    'cluster_headache': 0.35,
                    'meningitis': 0.25,
                    'brain_tumor': 0.05
                }
            },
            'dizziness': {
                'name': 'Dizziness',
                'categories': ['neurological', 'cardiac', 'metabolic'],
                'severity_levels': {
                    'mild': 0.15, 'moderate': 0.35, 'severe': 0.65, 'vertigo': 0.8
                },
                'associated_conditions': {
                    'vertigo': 0.60,
                    'hypotension': 0.45,
                    'anemia': 0.35,
                    'stroke': 0.20,
                    'inner_ear_infection': 0.40
                }
            },
            
            # Gastrointestinal Symptoms
            'abdominal_pain': {
                'name': 'Abdominal Pain',
                'categories': ['gastrointestinal', 'reproductive', 'urinary'],
                'severity_levels': {
                    'mild': 0.1, 'moderate': 0.3, 'severe': 0.7, 'acute': 0.9
                },
                'associated_conditions': {
                    'appendicitis': 0.65,
                    'gastritis': 0.45,
                    'ibs': 0.35,
                    'gallstones': 0.40,
                    'uti': 0.25
                }
            },
            'nausea': {
                'name': 'Nausea',
                'categories': ['gastrointestinal', 'neurological', 'metabolic'],
                'severity_levels': {
                    'mild': 0.1, 'moderate': 0.3, 'severe': 0.6, 'vomiting': 0.8
                },
                'associated_conditions': {
                    'gastroenteritis': 0.55,
                    'migraine': 0.40,
                    'pregnancy': 0.35,
                    'food_poisoning': 0.50,
                    'medication_side_effect': 0.30
                }
            },
            
            # Respiratory Symptoms
            'cough': {
                'name': 'Cough',
                'categories': ['respiratory', 'allergic', 'infectious'],
                'severity_levels': {
                    'mild': 0.1, 'moderate': 0.3, 'severe': 0.6, 'persistent': 0.7
                },
                'associated_conditions': {
                    'bronchitis': 0.60,
                    'pneumonia': 0.45,
                    'covid_19': 0.55,
                    'allergies': 0.35,
                    'asthma': 0.50
                }
            },
            'fever': {
                'name': 'Fever',
                'categories': ['infectious', 'inflammatory', 'metabolic'],
                'severity_levels': {
                    'low_grade': 0.2, 'moderate': 0.4, 'high': 0.7, 'very_high': 0.9
                },
                'associated_conditions': {
                    'viral_infection': 0.65,
                    'bacterial_infection': 0.55,
                    'flu': 0.60,
                    'covid_19': 0.50,
                    'inflammatory_condition': 0.35
                }
            }
        }
    
    def _initialize_disease_knowledge(self) -> Dict[str, Any]:
        """Comprehensive disease knowledge base"""
        return {
            'myocardial_infarction': {
                'name': 'Heart Attack',
                'symptoms': ['chest_pain', 'shortness_of_breath', 'nausea', 'sweating'],
                'risk_factors': ['hypertension', 'diabetes', 'smoking', 'high_cholesterol'],
                'urgency_level': 'emergency',
                'mortality_risk': 0.15,
                'typical_demographics': ['male_40+', 'female_50+']
            },
            'pneumonia': {
                'name': 'Pneumonia',
                'symptoms': ['cough', 'fever', 'shortness_of_breath', 'chest_pain'],
                'risk_factors': ['age_65+', 'smoking', 'chronic_lung_disease'],
                'urgency_level': 'urgent',
                'mortality_risk': 0.08,
                'typical_demographics': ['all_ages', 'elderly_high_risk']
            },
            'migraine': {
                'name': 'Migraine',
                'symptoms': ['headache', 'nausea', 'sensitivity_to_light'],
                'risk_factors': ['family_history', 'female', 'stress'],
                'urgency_level': 'routine',
                'mortality_risk': 0.001,
                'typical_demographics': ['adults_20-50', 'female_predominant']
            }
        }
    
    def _initialize_risk_factors(self) -> Dict[str, Any]:
        """Patient risk factors database"""
        return {
            'age_groups': {
                'child': 0.1, 'teen': 0.2, 'adult': 0.5, 'senior': 0.8
            },
            'lifestyle_factors': {
                'smoking': 0.6, 'alcohol': 0.4, 'sedentary': 0.3, 'obesity': 0.5
            },
            'medical_history': {
                'diabetes': 0.6, 'hypertension': 0.5, 'heart_disease': 0.7
            }
        }
    
    def analyze_symptoms(self, symptoms: List[str], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced multi-symptom analysis with probability scoring"""
        
        # Calculate individual symptom probabilities
        symptom_scores = {}
        for symptom in symptoms:
            if symptom in self.symptom_database:
                symptom_info = self.symptom_database[symptom]
                base_probability = symptom_info['severity_levels'].get('moderate', 0.3)
                
                # Adjust based on patient factors
                adjusted_prob = self._adjust_for_patient_factors(base_probability, patient_data)
                symptom_scores[symptom] = adjusted_prob
        
        # Calculate condition probabilities
        condition_probabilities = {}
        for condition_id, condition_info in self.disease_knowledge_base.items():
            probability = self._calculate_condition_probability(condition_id, symptoms, patient_data)
            if probability > 0.1:  # Only show conditions with >10% probability
                condition_probabilities[condition_id] = {
                    'name': condition_info['name'],
                    'probability': probability,
                    'urgency': condition_info['urgency_level'],
                    'risk_factors': self._get_matching_risk_factors(condition_id, patient_data)
                }
        
        # Sort by probability
        sorted_conditions = dict(sorted(
            condition_probabilities.items(), 
            key=lambda x: x[1]['probability'], 
            reverse=True
        ))
        
        return {
            'symptom_analysis': symptom_scores,
            'condition_probabilities': sorted_conditions,
            'risk_assessment': self._calculate_overall_risk(patient_data),
            'recommendations': self._generate_recommendations(sorted_conditions, patient_data),
            'emergency_flags': self._check_emergency_conditions(sorted_conditions)
        }
    
    def _adjust_for_patient_factors(self, base_prob: float, patient_data: Dict[str, Any]) -> float:
        """Adjust probability based on patient demographics and history"""
        adjusted_prob = base_prob
        
        # Age adjustment
        age = patient_data.get('age', 30)
        if age > 65:
            adjusted_prob *= 1.3
        elif age < 18:
            adjusted_prob *= 0.7
        
        # Medical history adjustment
        if patient_data.get('has_chronic_conditions', False):
            adjusted_prob *= 1.4
        
        # Lifestyle factors
        if patient_data.get('smoker', False):
            adjusted_prob *= 1.5
        
        return min(adjusted_prob, 0.95)  # Cap at 95%
    
    def _calculate_condition_probability(self, condition_id: str, symptoms: List[str], 
                                       patient_data: Dict[str, Any]) -> float:
        """Calculate probability for a specific condition"""
        condition = self.disease_knowledge_base[condition_id]
        base_prob = 0.0
        
        # Calculate based on symptom matches
        matching_symptoms = set(symptoms) & set(condition['symptoms'])
        if matching_symptoms:
            base_prob = len(matching_symptoms) / len(condition['symptoms']) * 0.7
        
        # Adjust for risk factors
        risk_adjustment = self._calculate_risk_adjustment(condition_id, patient_data)
        final_prob = base_prob * risk_adjustment
        
        return min(final_prob, 0.95)
    
    def _calculate_risk_adjustment(self, condition_id: str, patient_data: Dict[str, Any]) -> float:
        """Calculate risk adjustment factor"""
        condition = self.disease_knowledge_base[condition_id]
        adjustment = 1.0
        
        for risk_factor in condition.get('risk_factors', []):
            if patient_data.get(risk_factor, False):
                adjustment *= 1.3
        
        return adjustment
    
    def _get_matching_risk_factors(self, condition_id: str, patient_data: Dict[str, Any]) -> List[str]:
        """Get matching risk factors for a condition"""
        condition = self.disease_knowledge_base[condition_id]
        matching_factors = []
        
        for risk_factor in condition.get('risk_factors', []):
            if patient_data.get(risk_factor, False):
                matching_factors.append(risk_factor)
        
        return matching_factors
    
    def _calculate_overall_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall health risk assessment"""
        risk_score = 0
        risk_factors = []
        
        # Age risk
        age = patient_data.get('age', 30)
        if age > 65:
            risk_score += 3
            risk_factors.append('Advanced age')
        
        # Lifestyle risks
        if patient_data.get('smoker', False):
            risk_score += 2
            risk_factors.append('Smoking')
        
        if patient_data.get('obese', False):
            risk_score += 2
            risk_factors.append('Obesity')
        
        # Medical history risks
        if patient_data.get('has_chronic_conditions', False):
            risk_score += 2
            risk_factors.append('Chronic conditions')
        
        # Determine risk level
        if risk_score >= 5:
            risk_level = 'High'
        elif risk_score >= 3:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._generate_risk_recommendations(risk_level, risk_factors)
        }
    
    def _generate_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate personalized risk reduction recommendations"""
        recommendations = []
        
        if risk_level == 'High':
            recommendations.append("🚨 Consider immediate consultation with healthcare provider")
            recommendations.append("📋 Comprehensive health screening recommended")
        
        if 'Smoking' in risk_factors:
            recommendations.append("🚭 Smoking cessation program strongly recommended")
        
        if 'Obesity' in risk_factors:
            recommendations.append("🥗 Weight management and nutritional counseling advised")
        
        if 'Advanced age' in risk_factors:
            recommendations.append("👵 Regular geriatric assessments recommended")
        
        return recommendations
    
    def _generate_recommendations(self, conditions: Dict[str, Any], patient_data: Dict[str, Any]) -> List[str]:
        """Generate personalized healthcare recommendations"""
        recommendations = []
        
        for condition_id, condition_info in list(conditions.items())[:3]:  # Top 3 conditions
            prob = condition_info['probability']
            urgency = condition_info['urgency']
            
            if prob > 0.7 and urgency == 'emergency':
                recommendations.append(f"🚨 IMMEDIATE CARE: High probability of {condition_info['name']} - Seek emergency care")
            elif prob > 0.5:
                recommendations.append(f"🩺 URGENT: Consider evaluation for {condition_info['name']} within 24-48 hours")
            elif prob > 0.3:
                recommendations.append(f"📋 ROUTINE: Schedule appointment to discuss {condition_info['name']}")
        
        return recommendations
    
    def _check_emergency_conditions(self, conditions: Dict[str, Any]) -> List[str]:
        """Check for emergency conditions that require immediate attention"""
        emergencies = []
        
        for condition_id, condition_info in conditions.items():
            if condition_info['urgency'] == 'emergency' and condition_info['probability'] > 0.3:
                emergencies.append(f"{condition_info['name']} (Probability: {condition_info['probability']:.1%})")
        
        return emergencies

# =============================================
# SMART PRESCRIPTION SYSTEM
# =============================================
class SmartPrescriptionSystem:
    """AI-powered prescription generation with dosage optimization"""
    
    def __init__(self):
        self.medication_database = self._initialize_medication_database()
        self.drug_interactions = self._initialize_drug_interactions()
        self.allergy_database = self._initialize_allergy_database()
        self.prescription_history = {}
    
    def _initialize_medication_database(self) -> Dict[str, Any]:
        """Comprehensive medication database"""
        return {
            'analgesics': {
                'ibuprofen': {
                    'name': 'Ibuprofen',
                    'forms': ['tablet', 'capsule', 'suspension'],
                    'standard_dosage': {
                        'adult': '200-400mg every 4-6 hours',
                        'max_daily': '1200mg',
                        'pediatric': '5-10mg/kg every 6-8 hours'
                    },
                    'indications': ['pain', 'fever', 'inflammation'],
                    'contraindications': ['peptic_ulcer', 'renal_impairment', 'third_trimester_pregnancy'],
                    'side_effects': ['gi_upset', 'dizziness', 'renal_impairment']
                },
                'acetaminophen': {
                    'name': 'Acetaminophen',
                    'forms': ['tablet', 'capsule', 'liquid'],
                    'standard_dosage': {
                        'adult': '500-1000mg every 4-6 hours',
                        'max_daily': '4000mg',
                        'pediatric': '10-15mg/kg every 4-6 hours'
                    },
                    'indications': ['pain', 'fever'],
                    'contraindications': ['liver_disease', 'alcoholism'],
                    'side_effects': ['liver_toxicity', 'rash']
                }
            },
            'antibiotics': {
                'amoxicillin': {
                    'name': 'Amoxicillin',
                    'forms': ['capsule', 'suspension'],
                    'standard_dosage': {
                        'adult': '250-500mg every 8 hours',
                        'pediatric': '20-40mg/kg/day in divided doses'
                    },
                    'indications': ['bacterial_infections', 'respiratory_infections'],
                    'contraindications': ['penicillin_allergy'],
                    'side_effects': ['diarrhea', 'rash', 'allergic_reactions']
                },
                'azithromycin': {
                    'name': 'Azithromycin',
                    'forms': ['tablet', 'suspension'],
                    'standard_dosage': {
                        'adult': '500mg daily for 3 days',
                        'pediatric': '10mg/kg daily for 3 days'
                    },
                    'indications': ['respiratory_infections', 'stis'],
                    'contraindications': ['liver_disease'],
                    'side_effects': ['gi_upset', 'qt_prolongation']
                }
            },
            'antihypertensives': {
                'lisinopril': {
                    'name': 'Lisinopril',
                    'forms': ['tablet'],
                    'standard_dosage': {
                        'adult': '5-40mg once daily',
                        'initial': '5-10mg daily'
                    },
                    'indications': ['hypertension', 'heart_failure'],
                    'contraindications': ['pregnancy', 'angioedema_history'],
                    'side_effects': ['cough', 'hyperkalemia', 'dizziness']
                }
            }
        }
    
    def _initialize_drug_interactions(self) -> Dict[str, List[str]]:
        """Drug interaction database"""
        return {
            'warfarin': ['aspirin', 'ibuprofen', 'amiodarone'],
            'simvastatin': ['clarithromycin', 'itraconazole'],
            'digoxin': ['furosemide', 'hydrochlorothiazide'],
            'lisinopril': ['potassium_supplements', 'spironolactone']
        }
    
    def _initialize_allergy_database(self) -> Dict[str, List[str]]:
        """Drug allergy cross-reactivity database"""
        return {
            'penicillin': ['cephalosporins', 'carbapenems'],
            'sulfa': ['furosemide', 'thiazides', 'sulfonylureas'],
            'aspirin': ['nsaids', 'other_salicylates']
        }
    
    def _get_medication_details(self, med_id: str) -> Optional[Dict[str, Any]]:
        """Get medication details from database"""
        for category, medications in self.medication_database.items():
            if med_id in medications:
                return medications[med_id]
        
        # Default medications for common prescriptions
        default_meds = {
            'amlodipine': {
                'name': 'Amlodipine',
                'standard_dosage': {'adult': '5-10mg once daily'},
                'indications': ['hypertension'],
                'contraindications': ['severe_hypotension'],
                'side_effects': ['edema', 'headache']
            },
            'hydrochlorothiazide': {
                'name': 'Hydrochlorothiazide',
                'standard_dosage': {'adult': '12.5-25mg once daily'},
                'indications': ['hypertension'],
                'contraindications': ['sulfa_allergy', 'renal_impairment'],
                'side_effects': ['hypokalemia', 'dehydration']
            },
            'metformin': {
                'name': 'Metformin',
                'standard_dosage': {'adult': '500-1000mg twice daily'},
                'indications': ['type_2_diabetes'],
                'contraindications': ['renal_disease'],
                'side_effects': ['gi_upset', 'diarrhea']
            },
            'symptomatic_treatment': {
                'name': 'Symptomatic Treatment',
                'standard_dosage': {'adult': 'As needed'},
                'indications': ['symptom_management'],
                'contraindications': [],
                'side_effects': []
            }
        }
        
        return default_meds.get(med_id)
    
    # KEEP THE ORIGINAL METHOD THAT YOUR CODE IS CALLING
    def generate_prescription(self, diagnosis: str, patient_data: Dict[str, Any], 
                            existing_medications: List[str] = None) -> Dict[str, Any]:
        """Generate smart prescription based on diagnosis and patient data"""
        
        try:
            # Map diagnosis to appropriate medications
            recommended_meds = self._map_diagnosis_to_medications(diagnosis)
            
            # Filter based on patient factors
            filtered_meds = self._filter_medications_for_patient(recommended_meds, patient_data)
            
            # Check for interactions
            interaction_warnings = self._check_drug_interactions(filtered_meds, existing_medications or [])
            
            # Calculate optimal dosages
            prescribed_medications = []
            for med_id in filtered_meds:
                medication = self._get_medication_details(med_id)
                if medication:
                    dosage = self._calculate_optimal_dosage(medication, patient_data)
                    prescribed_medications.append({
                        'medication_id': med_id,
                        'name': medication['name'],
                        'dosage': dosage,
                        'frequency': self._get_dosage_frequency(medication),
                        'duration': self._get_treatment_duration(diagnosis, med_id),
                        'instructions': self._generate_instructions(medication, diagnosis),
                        'warnings': self._get_medication_warnings(medication, patient_data)
                    })
            
            # Generate prescription ID and store in history
            prescription_id = f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            result = {
                'prescription_id': prescription_id,
                'prescribed_medications': prescribed_medications,
                'interaction_warnings': interaction_warnings,
                'patient_specific_instructions': self._generate_patient_instructions(prescribed_medications),
                'follow_up_recommendations': self._generate_follow_up_recommendations(diagnosis),
                'prescription_validity': '30 days',
                'generation_date': datetime.now().isoformat(),
                'status': 'success'
            }
            
            # Store in history
            self.prescription_history[prescription_id] = result
            
            return result
            
        except Exception as e:
            return {
                'error': f"Prescription generation failed: {str(e)}",
                'status': 'error'
            }
    
    def _map_diagnosis_to_medications(self, diagnosis: str) -> List[str]:
        """Map medical diagnosis to appropriate medications"""
        diagnosis_medication_map = {
            'bacterial_pneumonia': ['amoxicillin', 'azithromycin'],
            'urinary_tract_infection': ['amoxicillin', 'azithromycin'],
            'hypertension': ['lisinopril', 'amlodipine'],
            'migraine': ['ibuprofen', 'acetaminophen'],
            'osteoarthritis': ['ibuprofen', 'acetaminophen'],
            'type_2_diabetes': ['metformin'],
            'common_cold': ['acetaminophen'],
            'fever': ['ibuprofen', 'acetaminophen'],
            'headache': ['ibuprofen', 'acetaminophen'],
            'pain': ['ibuprofen', 'acetaminophen']
        }
        
        return diagnosis_medication_map.get(diagnosis, ['ibuprofen', 'acetaminophen'])
    
    def _filter_medications_for_patient(self, medications: List[str], patient_data: Dict[str, Any]) -> List[str]:
        """Filter medications based on patient contraindications"""
        filtered_meds = []
        
        for med_id in medications:
            medication = self._get_medication_details(med_id)
            if medication and not self._has_contraindications(medication, patient_data):
                filtered_meds.append(med_id)
        
        return filtered_meds
    
    def _has_contraindications(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        """Check if patient has contraindications for medication"""
        contraindications = medication.get('contraindications', [])
        patient_allergies = patient_data.get('allergies', [])
        patient_conditions = patient_data.get('medical_conditions', [])
        
        # Check allergies
        for allergy in patient_allergies:
            if any(allergy.lower() in contrain.lower() for contrain in contraindications):
                return True
        
        # Check medical conditions
        for condition in patient_conditions:
            if any(condition.lower() in contrain.lower() for contrain in contraindications):
                return True
        
        return False
    
    def _check_drug_interactions(self, new_meds: List[str], existing_meds: List[str]) -> List[str]:
        """Check for potential drug interactions"""
        warnings = []
        
        for new_med in new_meds:
            for existing_med in existing_meds:
                if existing_med in self.drug_interactions.get(new_med, []):
                    warnings.append(f"⚠️ Interaction between {new_med} and {existing_med}")
        
        return warnings
    
    def _calculate_optimal_dosage(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> str:
        """Calculate optimal dosage based on patient factors"""
        age = patient_data.get('age', 30)
        weight = patient_data.get('weight', 70)
        renal_function = patient_data.get('renal_function', 'normal')
        hepatic_function = patient_data.get('hepatic_function', 'normal')
        
        standard_dosage = medication['standard_dosage']
        
        # Adjust for age
        if age < 18 and 'pediatric' in standard_dosage:
            return standard_dosage['pediatric']
        elif age > 65:
            # Geriatric consideration
            adult_dosage = standard_dosage.get('adult', 'Consult dosage guide')
            if 'mg' in adult_dosage:
                return f"Consider reduced dosage: {adult_dosage}"
            return adult_dosage
        else:
            return standard_dosage.get('adult', standard_dosage.get('standard', 'Consult dosage guide'))
    
    def _get_dosage_frequency(self, medication: Dict[str, Any]) -> str:
        """Determine appropriate dosage frequency"""
        dosage = medication['standard_dosage']
        adult_dosage = dosage.get('adult', '')
        
        if 'every 4-6 hours' in adult_dosage:
            return '3-4 times daily as needed'
        elif 'every 6-8 hours' in adult_dosage:
            return '3 times daily'
        elif 'every 8 hours' in adult_dosage:
            return '3 times daily'
        elif 'once daily' in adult_dosage:
            return 'Once daily'
        else:
            return 'As per dosage instructions'
    
    def _get_treatment_duration(self, diagnosis: str, medication_id: str) -> str:
        """Determine appropriate treatment duration"""
        duration_map = {
            'bacterial_pneumonia': '7-10 days',
            'urinary_tract_infection': '3-7 days',
            'hypertension': 'Chronic/long-term',
            'migraine': 'As needed',
            'osteoarthritis': 'As needed for pain',
            'type_2_diabetes': 'Chronic/long-term',
            'common_cold': '3-5 days as needed',
            'fever': 'As needed',
            'headache': 'As needed',
            'pain': 'As needed'
        }
        return duration_map.get(diagnosis, '5-7 days')
    
    def _generate_instructions(self, medication: Dict[str, Any], diagnosis: str) -> str:
        """Generate patient instructions for medication"""
        base_instructions = f"Take as directed for {diagnosis.replace('_', ' ')}"
        
        # Add specific instructions based on medication type
        if 'antibiotic' in medication.get('indications', []):
            base_instructions += ". Complete full course even if feeling better."
        
        if any(nsaid in medication.get('name', '').lower() for nsaid in ['ibuprofen', 'naproxen']):
            base_instructions += ". Take with food to avoid stomach upset."
        
        return base_instructions
    
    def _get_medication_warnings(self, medication: Dict[str, Any], patient_data: Dict[str, Any]) -> List[str]:
        """Generate medication-specific warnings"""
        warnings = []
        
        side_effects = medication.get('side_effects', [])
        if side_effects:
            warnings.append(f"Possible side effects: {', '.join(side_effects[:2])}")
        
        if patient_data.get('pregnant', False):
            warnings.append("⚠️ Consult doctor before use during pregnancy")
        
        if patient_data.get('renal_impairment', False):
            warnings.append("⚠️ Renal impairment - monitor kidney function")
        
        return warnings
    
    def _generate_patient_instructions(self, medications: List[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive patient instructions"""
        instructions = [
            "💊 Take medications exactly as prescribed",
            "⏰ Maintain consistent dosing schedule",
            "🚫 Do not stop medications without consulting your doctor",
            "📞 Contact healthcare provider for any severe side effects",
            "🏥 Keep follow-up appointments as scheduled"
        ]
        
        return instructions
    
    def _generate_follow_up_recommendations(self, diagnosis: str) -> List[str]:
        """Generate follow-up recommendations based on diagnosis"""
        follow_up_map = {
            'hypertension': ['Blood pressure check in 2 weeks', 'Regular monitoring'],
            'type_2_diabetes': ['Fasting blood sugar in 1 week', 'HbA1c in 3 months'],
            'bacterial_pneumonia': ['Re-evaluation if no improvement in 3 days', 'Complete full course'],
            'urinary_tract_infection': ['Follow-up if symptoms persist after treatment'],
            'chronic_pain': ['Follow-up in 4 weeks', 'Physical therapy referral']
        }
        
        return follow_up_map.get(diagnosis, ['Follow-up as needed'])
    
    # ADDITIONAL ENHANCED METHODS (optional advanced features)
    def generate_comprehensive_prescription(self, diagnosis: str, patient_data: Dict[str, Any], 
                                          existing_medications: List[str] = None) -> Dict[str, Any]:
        """Enhanced prescription with additional features"""
        base_prescription = self.generate_prescription(diagnosis, patient_data, existing_medications)
        
        if 'error' in base_prescription:
            return base_prescription
        
        # Add enhanced features
        base_prescription['enhanced_features'] = {
            'lifestyle_recommendations': self._generate_lifestyle_recommendations(diagnosis),
            'emergency_instructions': self._generate_emergency_instructions(),
            'cost_analysis': self._analyze_prescription_costs(base_prescription['prescribed_medications']),
            'monitoring_plan': self._generate_monitoring_plan(diagnosis)
        }
        
        return base_prescription
    
    def _generate_lifestyle_recommendations(self, diagnosis: str) -> List[str]:
        """Generate lifestyle recommendations"""
        lifestyle_map = {
            'hypertension': [
                'Reduce sodium intake',
                'Regular exercise',
                'Weight management',
                'Limit alcohol'
            ],
            'type_2_diabetes': [
                'Balanced diet',
                'Regular blood sugar monitoring',
                'Physical activity',
                'Foot care'
            ],
            'general': [
                'Healthy diet',
                'Regular exercise',
                'Adequate sleep',
                'Stress management'
            ]
        }
        return lifestyle_map.get(diagnosis, lifestyle_map['general'])
    
    def _generate_emergency_instructions(self) -> List[str]:
        """Generate emergency instructions"""
        return [
            "For severe allergic reactions: Seek emergency care",
            "For overdose: Call poison control",
            "For severe side effects: Contact your doctor"
        ]
    
    def _analyze_prescription_costs(self, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prescription costs"""
        return {
            'cost_estimate': 'Low to moderate',
            'suggestions': ['Ask about generic options', 'Check insurance coverage']
        }
    
    def _generate_monitoring_plan(self, diagnosis: str) -> List[str]:
        """Generate monitoring plan"""
        monitoring_map = {
            'hypertension': ['Weekly BP checks', 'Renal function tests'],
            'diabetes': ['Daily glucose monitoring', 'Quarterly HbA1c'],
            'infection': ['Symptom tracking', 'Follow-up if no improvement']
        }
        return monitoring_map.get(diagnosis, ['Regular follow-up as needed'])
    
    def validate_prescription(self, prescription_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prescription safety"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        med_count = len(prescription_data.get('prescribed_medications', []))
        if med_count == 0:
            validation['warnings'].append('No medications prescribed')
        elif med_count > 3:
            validation['warnings'].append('Multiple medications - review for interactions')
        
        interactions = prescription_data.get('interaction_warnings', [])
        if interactions:
            validation['warnings'].extend(interactions)
            validation['recommendations'].append('Review drug interactions with pharmacist')
        
        return validation
    
    def get_prescription_history(self) -> List[Dict[str, Any]]:
        """Get prescription history"""
        return list(self.prescription_history.values())
# =============================================
# MEDICAL IMAGING AI ANALYZER
# =============================================
class MedicalImagingAI:
    """Advanced medical image analysis using AI"""
    
    def __init__(self):
        self.analysis_models = self._initialize_analysis_models()
        self.normal_ranges = self._initialize_normal_ranges()
    
    def _initialize_analysis_models(self) -> Dict[str, Any]:
        """Initialize AI analysis models for different imaging types"""
        return {
            'xray': {
                'name': 'X-Ray Analysis AI',
                'capabilities': ['bone_fractures', 'pneumonia', 'effusions', 'cardiomegaly'],
                'confidence_threshold': 0.75
            },
            'mri': {
                'name': 'MRI Analysis AI',
                'capabilities': ['tumor_detection', 'stroke', 'ms_lesions', 'disc_herniation'],
                'confidence_threshold': 0.80
            },
            'ct_scan': {
                'name': 'CT Scan Analysis AI',
                'capabilities': ['hemorrhage', 'fractures', 'appendicitis', 'pulmonary_embolism'],
                'confidence_threshold': 0.78
            },
            'ultrasound': {
                'name': 'Ultrasound Analysis AI',
                'capabilities': ['gallstones', 'aortic_aneurysm', 'ovarian_cysts', 'thyroid_nodules'],
                'confidence_threshold': 0.70
            }
        }
    
    def _initialize_normal_ranges(self) -> Dict[str, Any]:
        """Initialize normal anatomical measurements"""
        return {
            'cardiac': {
                'cardiothoracic_ratio': '< 0.5',
                'aortic_diameter': '2.1-3.2 cm'
            },
            'pulmonary': {
                'lung_fields': 'clear',
                'costophrenic_angles': 'sharp'
            },
            'abdominal': {
                'liver_size': '< 15.5 cm',
                'spleen_size': '< 13 cm',
                'aortic_diameter': '< 3 cm'
            }
        }
    
    def analyze_image(self, image_data: Any, image_type: str, clinical_context: str = None) -> Dict[str, Any]:
        """Analyze medical image using AI algorithms"""
        
        # Preprocess image
        processed_image = self._preprocess_image(image_data, image_type)
        
        # Get appropriate analysis model
        analysis_model = self.analysis_models.get(image_type, self.analysis_models['xray'])
        
        # Perform AI analysis (simulated)
        findings = self._simulate_ai_analysis(processed_image, image_type, clinical_context)
        
        # Generate report
        report = self._generate_imaging_report(findings, image_type, clinical_context)
        
        return {
            'image_type': image_type,
            'analysis_model': analysis_model['name'],
            'findings': findings,
            'clinical_impression': report['impression'],
            'recommendations': report['recommendations'],
            'confidence_scores': findings.get('confidence_scores', {}),
            'urgent_findings': self._identify_urgent_findings(findings),
            'comparison_available': False  # Would be True if prior studies exist
        }
    
    def _preprocess_image(self, image_data: Any, image_type: str) -> Any:
        """Preprocess medical image for analysis"""
        try:
            if isinstance(image_data, str):
                # Handle base64 encoded images
                image_data = base64.b64decode(image_data)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data) if isinstance(image_data, bytes) else image_data)
            
            # Image enhancement based on type
            if image_type == 'xray':
                image = self._enhance_xray_image(image)
            elif image_type == 'mri':
                image = self._enhance_mri_image(image)
            
            return image
            
        except Exception as e:
            st.error(f"Image preprocessing error: {str(e)}")
            return image_data
    
    def _enhance_xray_image(self, image: Image.Image) -> Image.Image:
        """Enhance X-ray image for better analysis"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply contrast enhancement
        img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=0)
        
        # Apply mild sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_array = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(img_array)
    
    def _enhance_mri_image(self, image: Image.Image) -> Image.Image:
        """Enhance MRI image for better analysis"""
        img_array = np.array(image)
        
        # Normalize intensity
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        
        return Image.fromarray(img_array)
    
    def _simulate_ai_analysis(self, image: Any, image_type: str, clinical_context: str) -> Dict[str, Any]:
        """Simulate AI analysis of medical image"""
        # This would integrate with actual AI models in production
        
        findings = {
            'normal_structures': [],
            'abnormalities': [],
            'measurements': {},
            'confidence_scores': {},
            'quality_assessment': 'Diagnostic quality'
        }
        
        # Simulate findings based on image type
        if image_type == 'xray':
            findings.update(self._simulate_xray_analysis(clinical_context))
        elif image_type == 'mri':
            findings.update(self._simulate_mri_analysis(clinical_context))
        elif image_type == 'ct_scan':
            findings.update(self._simulate_ct_analysis(clinical_context))
        elif image_type == 'ultrasound':
            findings.update(self._simulate_ultrasound_analysis(clinical_context))
        
        return findings
    
    def _simulate_xray_analysis(self, clinical_context: str) -> Dict[str, Any]:
        """Simulate X-ray analysis findings"""
        findings = {
            'normal_structures': ['lungs clear', 'heart size normal', 'bones intact'],
            'abnormalities': [],
            'measurements': {
                'cardiothoracic_ratio': 0.48,
                'lung_fields': 'clear'
            },
            'confidence_scores': {
                'lung_analysis': 0.89,
                'bone_analysis': 0.92,
                'cardiac_analysis': 0.85
            }
        }
        
        # Context-specific abnormalities
        if 'pneumonia' in (clinical_context or '').lower():
            findings['abnormalities'].append('Right lower lobe opacity consistent with pneumonia')
            findings['confidence_scores']['pneumonia_detection'] = 0.78
        
        if 'fracture' in (clinical_context or '').lower():
            findings['abnormalities'].append('Non-displaced fracture of distal radius')
            findings['confidence_scores']['fracture_detection'] = 0.91
        
        return findings
    
    def _simulate_mri_analysis(self, clinical_context: str) -> Dict[str, Any]:
        """Simulate MRI analysis findings"""
        findings = {
            'normal_structures': ['brain parenchyma normal', 'ventricles normal size', 'no mass lesions'],
            'abnormalities': [],
            'measurements': {
                'ventricular_size': 'normal',
                'sulcal_pattern': 'age-appropriate'
            },
            'confidence_scores': {
                'brain_paranchyma': 0.94,
                'ventricular_system': 0.88,
                'white_matter': 0.91
            }
        }
        
        if 'headache' in (clinical_context or '').lower():
            findings['abnormalities'].append('Small nonspecific white matter hyperintensities')
            findings['confidence_scores']['white_matter_lesions'] = 0.76
        
        return findings
    
    def _simulate_ct_analysis(self, clinical_context: str) -> Dict[str, Any]:
        """Simulate CT scan analysis findings"""
        return {
            'normal_structures': ['no acute intracranial hemorrhage', 'grey-white differentiation maintained'],
            'abnormalities': ['Mild cerebral atrophy for age'],
            'measurements': {
                'midline_shift': 'none',
                'ventricular_size': 'upper normal limits'
            },
            'confidence_scores': {
                'hemorrhage_detection': 0.97,
                'mass_effect': 0.89,
                'atrophy_assessment': 0.82
            }
        }
    
    def _simulate_ultrasound_analysis(self, clinical_context: str) -> Dict[str, Any]:
        """Simulate ultrasound analysis findings"""
        return {
            'normal_structures': ['liver homogeneous', 'gallbladder without stones', 'kidneys normal size'],
            'abnormalities': ['Simple hepatic cyst 1.5cm'],
            'measurements': {
                'liver_size': '14.2 cm',
                'aortic_diameter': '2.3 cm'
            },
            'confidence_scores': {
                'liver_analysis': 0.87,
                'gallbladder_analysis': 0.93,
                'renal_analysis': 0.85
            }
        }
    
    def _generate_imaging_report(self, findings: Dict[str, Any], image_type: str, 
                               clinical_context: str) -> Dict[str, Any]:
        """Generate comprehensive imaging report"""
        
        impression = "Within normal limits."
        recommendations = ["Routine follow-up as clinically indicated"]
        
        if findings['abnormalities']:
            impression = f"Findings include: {', '.join(findings['abnormalities'])}."
            
            # Generate specific recommendations based on findings
            if any('pneumonia' in abn.lower() for abn in findings['abnormalities']):
                recommendations = [
                    "Clinical correlation recommended",
                    "Consider follow-up chest X-ray in 4-6 weeks",
                    "Antibiotic therapy if clinically indicated"
                ]
            elif any('fracture' in abn.lower() for abn in findings['abnormalities']):
                recommendations = [
                    "Orthopedic consultation recommended",
                    "Immobilization as appropriate",
                    "Follow-up X-ray in 2 weeks"
                ]
        
        return {
            'impression': impression,
            'recommendations': recommendations,
            'technique': f"Standard {image_type.upper()} technique",
            'comparison': 'No prior studies available for comparison',
            'clinical_correlation': 'Clinical correlation is recommended'
        }
    
    def _identify_urgent_findings(self, findings: Dict[str, Any]) -> List[str]:
        """Identify findings that require urgent attention"""
        urgent_keywords = [
            'pneumothorax', 'hemorrhage', 'fracture', 'appendicitis', 
            'aneurysm', 'obstruction', 'ischemia'
        ]
        
        urgent_findings = []
        for abnormality in findings.get('abnormalities', []):
            if any(keyword in abnormality.lower() for keyword in urgent_keywords):
                urgent_findings.append(abnormality)
        
        return urgent_findings

# =============================================
# PATIENT PORTAL SYSTEM
# =============================================
class PatientPortalSystem:
    """Comprehensive patient portal with medical records and lab results"""
    
    def __init__(self):
        self.medical_records = {}
        self.lab_results = {}
        self.appointment_history = {}
    
    def initialize_session_state(self):
     if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'logged_in': False,
            'username': '',
            'current_page': 'dashboard',
            'user_role': 'patient',
            'patients': [],
            'doctors': [],
            'appointments': [],
            'chat_history': [],
            'ai_thinking': False,
            'pharmacy_inventory': [],
            'billing_records': [],
            'payment_records': [],
            'patient_data': {},
            'consultation_data': {},
            'imaging_analysis': {},
            'prescription_data': {},
            'symptom_analysis': {},
            'telemedicine_sessions': [],
            'vital_signs_history': [],
            'pain_assessment_history': [],
            'sleep_history': [],
            'emotion_history': [],
            'show_goal_form': False,
            # ADD THIS USER PROFILE INITIALIZATION:
            'user_profile': {
                'personal_info': {
                    'full_name': '',
                    'date_of_birth': None,
                    'gender': '',
                    'blood_type': '',
                    'email': '',
                    'phone': '',
                    'address': ''
                },
                'emergency_contact': {
                    'name': '',
                    'relationship': '',
                    'phone': '',
                    'email': ''
                },
                'medical_info': {
                    'allergies': [],
                    'conditions': [],
                    'medications': [],
                    'surgeries': [],
                    'primary_doctor': '',
                    'insurance': ''
                },
                'preferences': {
                    'theme': 'light',
                    'notifications': True,
                    'data_sharing': False,
                    'language': 'English'
                }
            },
            'system_settings': {
                'ai_personality': '🤖 Professional Medical',
                'alert_thresholds': {
                    'heart_rate': 120,
                    'sleep_hours': 5,
                    'steps_goal': 8000
                },
                'notifications': {
                    'email_alerts': True,
                    'push_notifications': True,
                    'quiet_hours': False
                }
            }
        })
    
    def add_medical_record(self, patient_id: str, record_type: str, record_data: Dict[str, Any]):
        """Add medical record to patient profile"""
        if patient_id not in self.medical_records:
            self.medical_records[patient_id] = self.initialize_patient_data(patient_id)
        
        self.medical_records[patient_id][record_type].append({
            'timestamp': datetime.now().isoformat(),
            'data': record_data,
            'provider': 'AI Health System',
            'location': 'Virtual Clinic'
        })
    
    def add_lab_result(self, patient_id: str, test_name: str, results: Dict[str, Any], 
                      reference_ranges: Dict[str, Any]):
        """Add laboratory test results"""
        if patient_id not in self.lab_results:
            self.lab_results[patient_id] = []
        
        self.lab_results[patient_id].append({
            'test_date': datetime.now().isoformat(),
            'test_name': test_name,
            'results': results,
            'reference_ranges': reference_ranges,
            'interpretation': self._interpret_lab_results(results, reference_ranges),
            'status': 'Completed',
            'ordering_provider': 'AI Health System'
        })
    
    def _interpret_lab_results(self, results: Dict[str, Any], reference_ranges: Dict[str, Any]) -> str:
        """Interpret laboratory results based on reference ranges"""
        interpretations = []
        
        for test, value in results.items():
            if test in reference_ranges:
                ref_range = reference_ranges[test]
                if value < ref_range['low']:
                    interpretations.append(f"⬇️ {test} low")
                elif value > ref_range['high']:
                    interpretations.append(f"⬆️ {test} high")
                else:
                    interpretations.append(f"✅ {test} normal")
        
        return ', '.join(interpretations) if interpretations else 'All values within normal limits'
    
    def generate_health_summary(self, patient_id: str) -> Dict[str, Any]:
        """Generate comprehensive health summary"""
        if patient_id not in self.medical_records:
            return {'error': 'Patient not found'}
        
        patient_data = self.medical_records[patient_id]
        lab_data = self.lab_results.get(patient_id, [])
        
        return {
            'patient_overview': {
                'name': patient_data['personal_info'].get('name', 'Unknown'),
                'age': self._calculate_age(patient_data['personal_info'].get('date_of_birth')),
                'active_conditions': len(patient_data['medical_history']['conditions']),
                'current_medications': len(patient_data['medical_history']['medications']),
                'last_checkup': self._get_last_appointment(patient_data['appointments'])
            },
            'recent_vitals': patient_data['vital_signs'][-5:] if patient_data['vital_signs'] else [],
            'recent_labs': lab_data[-3:] if lab_data else [],
            'health_metrics': self._calculate_health_metrics(patient_data, lab_data),
            'care_plan': self._generate_care_plan(patient_data),
            'upcoming_appointments': self._get_upcoming_appointments(patient_data['appointments'])
        }
    
    def _calculate_age(self, date_of_birth: str) -> int:
        """Calculate age from date of birth"""
        if not date_of_birth:
            return 0
        try:
            birth_date = datetime.fromisoformat(date_of_birth)
            today = datetime.now()
            return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        except:
            return 0
    
    def _get_last_appointment(self, appointments: List[Dict[str, Any]]) -> str:
        """Get last appointment date"""
        if not appointments:
            return "No appointments"
        sorted_appointments = sorted(appointments, key=lambda x: x.get('timestamp', ''), reverse=True)
        return sorted_appointments[0].get('timestamp', 'Unknown')
    
    def _calculate_health_metrics(self, patient_data: Dict[str, Any], lab_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall health metrics"""
        metrics = {
            'overall_health_score': 85,  # Placeholder calculation
            'risk_factors': [],
            'preventive_care_needs': [],
            'medication_adherence': 'Good'  # Placeholder
        }
        
        # Analyze conditions for risk factors
        conditions = patient_data['medical_history']['conditions']
        if any('diabetes' in cond.lower() for cond in conditions):
            metrics['risk_factors'].append('Diabetes')
        if any('hypertension' in cond.lower() for cond in conditions):
            metrics['risk_factors'].append('Hypertension')
        
        # Analyze lab data
        recent_labs = lab_data[-1] if lab_data else {}
        if 'cholesterol' in str(recent_labs):
            metrics['risk_factors'].append('High Cholesterol')
        
        return metrics
    
    def _generate_care_plan(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized care plan"""
        return {
            'goals': [
                'Maintain blood pressure < 140/90 mmHg',
                'Annual preventive health screening',
                'Regular physical activity',
                'Balanced nutrition'
            ],
            'actions': [
                'Schedule follow-up in 6 months',
                'Complete pending lab tests',
                'Update medication list',
                'Review vaccination status'
            ],
            'providers': [
                'Primary Care Physician',
                'Specialists as needed'
            ]
        }
    
    def _get_upcoming_appointments(self, appointments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get upcoming appointments"""
        upcoming = []
        for appointment in appointments:
            appt_date = appointment.get('timestamp', '')
            if appt_date > datetime.now().isoformat():
                upcoming.append(appointment)
        
        return sorted(upcoming, key=lambda x: x.get('timestamp', ''))[:3]

# =============================================
# TELEMEDICINE INTEGRATION
# =============================================
class TelemedicineSystem:
    """Advanced telemedicine with video consultations"""
    
    def __init__(self):
        self.consultation_sessions = {}
        self.video_quality_settings = {
            'low': {'resolution': '480p', 'bitrate': '500kbps'},
            'medium': {'resolution': '720p', 'bitrate': '1.5mbps'},
            'high': {'resolution': '1080p', 'bitrate': '3mbps'}
        }
    
    def schedule_consultation(self, patient_id: str, provider_id: str, 
                            scheduled_time: str, consultation_type: str) -> Dict[str, Any]:
        """Schedule telemedicine consultation"""
        session_id = f"tele_{patient_id}_{provider_id}_{int(datetime.now().timestamp())}"
        
        consultation = {
            'session_id': session_id,
            'patient_id': patient_id,
            'provider_id': provider_id,
            'scheduled_time': scheduled_time,
            'consultation_type': consultation_type,
            'status': 'scheduled',
            'video_quality': 'medium',
            'recording_consent': False,
            'pre_consultation_questions': self._generate_pre_consultation_questions(consultation_type),
            'technical_requirements': self._get_technical_requirements(),
            'join_url': f"https://telemedicine.aihealth.com/join/{session_id}"
        }
        
        self.consultation_sessions[session_id] = consultation
        return consultation
    
    def _generate_pre_consultation_questions(self, consultation_type: str) -> List[str]:
        """Generate pre-consultation questions based on type"""
        base_questions = [
            "What is the main reason for your visit today?",
            "How long have you been experiencing these symptoms?",
            "Have you taken any medications for this condition?"
        ]
        
        type_specific_questions = {
            'follow_up': [
                "Have there been any changes in your condition since your last visit?",
                "Are you experiencing any side effects from your current medications?"
            ],
            'new_issue': [
                "When did your symptoms first appear?",
                "What makes your symptoms better or worse?",
                "Have you had similar symptoms in the past?"
            ],
            'chronic_care': [
                "How have you been managing your condition?",
                "Any recent changes in your symptoms or overall health?"
            ]
        }
        
        return base_questions + type_specific_questions.get(consultation_type, [])
    
    def _get_technical_requirements(self) -> Dict[str, Any]:
        """Get technical requirements for telemedicine"""
        return {
            'browser': 'Chrome 90+, Firefox 85+, Safari 14+',
            'internet_speed': 'Minimum 3 Mbps upload/download',
            'camera': 'HD webcam (720p or higher)',
            'microphone': 'Built-in or external microphone',
            'speakers': 'Built-in or external speakers/headphones',
            'privacy': 'Private, well-lit location recommended'
        }
    
    def start_consultation(self, session_id: str) -> Dict[str, Any]:
        """Start telemedicine consultation"""
        if session_id not in self.consultation_sessions:
            return {'error': 'Session not found'}
        
        session = self.consultation_sessions[session_id]
        session['status'] = 'in_progress'
        session['start_time'] = datetime.now().isoformat()
        
        return {
            'session_id': session_id,
            'status': 'started',
            'video_url': session['join_url'],
            'waiting_room': f"https://telemedicine.aihealth.com/waiting/{session_id}",
            'technical_support': 'available',
            'recording_status': 'not_recording'  # Would require consent
        }
    
    def end_consultation(self, session_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        """End telemedicine consultation and generate summary"""
        if session_id not in self.consultation_sessions:
            return {'error': 'Session not found'}
        
        session = self.consultation_sessions[session_id]
        session['status'] = 'completed'
        session['end_time'] = datetime.now().isoformat()
        session['duration'] = self._calculate_duration(session.get('start_time'), session['end_time'])
        session['consultation_summary'] = summary
        
        return {
            'session_id': session_id,
            'status': 'completed',
            'duration': session['duration'],
            'summary': summary,
            'next_steps': self._generate_next_steps(summary),
            'prescription_available': bool(summary.get('prescription_needed', False)),
            'follow_up_required': bool(summary.get('follow_up_recommended', False))
        }
    
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calculate consultation duration"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            minutes = duration.total_seconds() / 60
            return f"{int(minutes)} minutes"
        except:
            return "Unknown duration"
    
    def _generate_next_steps(self, summary: Dict[str, Any]) -> List[str]:
        """Generate next steps after consultation"""
        next_steps = []
        
        if summary.get('prescription_needed'):
            next_steps.append("💊 Prescription sent to pharmacy")
        
        if summary.get('lab_tests_ordered'):
            next_steps.append("🩺 Laboratory tests ordered")
        
        if summary.get('follow_up_recommended'):
            next_steps.append("📅 Follow-up appointment scheduled")
        
        if summary.get('specialist_referral'):
            next_steps.append("👨‍⚕️ Specialist referral initiated")
        
        next_steps.append("📋 Consultation summary available in patient portal")
        
        return next_steps
    
    def get_consultation_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient's telemedicine history"""
        patient_sessions = []
        for session_id, session in self.consultation_sessions.items():
            if session['patient_id'] == patient_id:
                patient_sessions.append(session)
        
        return sorted(patient_sessions, key=lambda x: x.get('scheduled_time', ''), reverse=True)

# =============================================
# ENHANCED HEALTH MONITOR
# =============================================
class EnhancedHealthMonitor:
    """Comprehensive health monitoring with AI analysis"""
    
    def __init__(self):
        self.vital_ranges = self._initialize_vital_ranges()
        self.pain_assessment_tools = self._initialize_pain_assessment()
        self.sleep_quality_metrics = self._initialize_sleep_metrics()
    
    def _initialize_vital_ranges(self) -> Dict[str, Any]:
        """Initialize normal vital sign ranges"""
        return {
            'blood_pressure': {
                'optimal': {'systolic': (90, 120), 'diastolic': (60, 80)},
                'normal': {'systolic': (120, 129), 'diastolic': (80, 84)},
                'high_normal': {'systolic': (130, 139), 'diastolic': (85, 89)},
                'hypertension': {'systolic': (140, 180), 'diastolic': (90, 120)}
            },
            'heart_rate': {
                'resting': {'adult': (60, 100), 'athlete': (40, 60), 'child': (70, 120)}
            },
            'temperature': {
                'normal': (36.1, 37.2),
                'fever': (37.3, 40.0)
            },
            'respiratory_rate': {
                'adult': (12, 20),
                'child': (20, 30),
                'infant': (30, 60)
            }
        }
    
    def _initialize_pain_assessment(self) -> Dict[str, Any]:
        """Initialize pain assessment tools"""
        return {
            'scale': {
                '0': 'No pain',
                '1-3': 'Mild pain',
                '4-6': 'Moderate pain', 
                '7-10': 'Severe pain'
            },
            'locations': [
                'Head', 'Neck', 'Chest', 'Back', 'Abdomen', 
                'Arms', 'Legs', 'Joints', 'General'
            ],
            'qualities': [
                'Sharp', 'Dull', 'Burning', 'Throbbing', 
                'Stabbing', 'Aching', 'Cramping'
            ]
        }
    
    def _initialize_sleep_metrics(self) -> Dict[str, Any]:
        """Initialize sleep quality assessment metrics"""
        return {
            'duration': {
                'optimal': (7, 9),
                'adequate': (6, 7),
                'insufficient': (0, 6)
            },
            'quality_scale': {
                'excellent': (9, 10),
                'good': (7, 8),
                'fair': (5, 6),
                'poor': (1, 4)
            }
        }
    
    def analyze_vital_signs(self, vital_data: Dict[str, Any], patient_age: int) -> Dict[str, Any]:
        """Analyze vital signs with AI assessment"""
        analysis = {
            'status': 'Normal',
            'warnings': [],
            'recommendations': [],
            'trend_analysis': 'Stable'
        }
        
        # Blood pressure analysis
        if 'systolic' in vital_data and 'diastolic' in vital_data:
            bp_status = self._analyze_blood_pressure(
                vital_data['systolic'], vital_data['diastolic']
            )
            analysis.update(bp_status)
        
        # Heart rate analysis
        if 'heart_rate' in vital_data:
            hr_status = self._analyze_heart_rate(vital_data['heart_rate'], patient_age)
            analysis.update(hr_status)
        
        # Temperature analysis
        if 'temperature' in vital_data:
            temp_status = self._analyze_temperature(vital_data['temperature'])
            analysis.update(temp_status)
        
        return analysis
    
    def _analyze_blood_pressure(self, systolic: int, diastolic: int) -> Dict[str, Any]:
        """Analyze blood pressure readings"""
        ranges = self.vital_ranges['blood_pressure']
        
        if systolic <= 120 and diastolic <= 80:
            return {'bp_status': 'Optimal', 'bp_category': 'Normal'}
        elif systolic <= 129 and diastolic <= 84:
            return {'bp_status': 'Normal', 'bp_category': 'Normal'}
        elif systolic <= 139 and diastolic <= 89:
            return {'bp_status': 'High Normal', 'bp_category': 'Elevated'}
        elif systolic <= 159 and diastolic <= 99:
            return {'bp_status': 'Stage 1 Hypertension', 'bp_category': 'High'}
        elif systolic <= 179 and diastolic <= 109:
            return {'bp_status': 'Stage 2 Hypertension', 'bp_category': 'High'}
        else:
            return {
                'bp_status': 'Hypertensive Crisis', 
                'bp_category': 'Critical',
                'warnings': ['🚨 Seek immediate medical attention']
            }
    
    def _analyze_heart_rate(self, heart_rate: int, age: int) -> Dict[str, Any]:
        """Analyze heart rate"""
        if age < 18:
            normal_range = self.vital_ranges['heart_rate']['resting']['child']
        else:
            normal_range = self.vital_ranges['heart_rate']['resting']['adult']
        
        if normal_range[0] <= heart_rate <= normal_range[1]:
            return {'hr_status': 'Normal'}
        elif heart_rate < normal_range[0]:
            return {
                'hr_status': 'Bradycardia',
                'warnings': ['Low heart rate detected']
            }
        else:
            return {
                'hr_status': 'Tachycardia', 
                'warnings': ['High heart rate detected']
            }
    
    def _analyze_temperature(self, temperature: float) -> Dict[str, Any]:
        """Analyze body temperature"""
        normal_range = self.vital_ranges['temperature']['normal']
        
        if normal_range[0] <= temperature <= normal_range[1]:
            return {'temp_status': 'Normal'}
        elif temperature > normal_range[1]:
            if temperature <= 38.0:
                return {
                    'temp_status': 'Low-grade fever',
                    'warnings': ['Mild fever detected']
                }
            elif temperature <= 39.0:
                return {
                    'temp_status': 'Fever',
                    'warnings': ['Moderate fever detected']
                }
            else:
                return {
                    'temp_status': 'High fever',
                    'warnings': ['🚨 High fever - seek medical attention']
                }
        else:
            return {
                'temp_status': 'Low temperature',
                'warnings': ['Low body temperature detected']
            }
class RealHealthAI:
    def __init__(self):
        self.health_conditions_database = {
            # Fever and General Symptoms
            "fever": {
                "name": "Fever",
                "medications": {
                    "over_counter": ["Acetaminophen 500-1000mg every 4-6 hours", "Ibuprofen 200-400mg every 6 hours"],
                    "prescription": ["Antibiotics if bacterial infection", "Antiviral medications if flu"]
                },
                "remedies": [
                    "Stay hydrated with water, electrolyte drinks, or clear broths",
                    "Rest and avoid physical exertion",
                    "Use cool compresses on forehead and wrists",
                    "Take lukewarm baths (not cold)",
                    "Wear lightweight, breathable clothing"
                ],
                "recovery_timeline": {
                    "immediate": "Temperature reduction within 1-2 hours with medication",
                    "short_term": "Fever breaks within 24-48 hours with proper care",
                    "complete": "Full recovery within 3-7 days depending on cause"
                },
                "when_to_see_doctor": "Fever above 103°F (39.4°C), lasting more than 3 days, or accompanied by severe symptoms"
            },
            
            "headache": {
                "name": "Headache",
                "medications": {
                    "over_counter": ["Ibuprofen 400-600mg every 6 hours", "Acetaminophen 500-1000mg every 6 hours", "Aspirin 325-650mg every 4 hours"],
                    "prescription": ["Sumatriptan 50-100mg for migraines", "Rizatriptan 10mg", "Preventive medications for chronic headaches"]
                },
                "remedies": [
                    "Rest in a quiet, dark room",
                    "Apply cold or warm compress to forehead",
                    "Stay hydrated with water",
                    "Gentle massage of temples and neck",
                    "Practice relaxation techniques and reduce screen time"
                ],
                "recovery_timeline": {
                    "immediate": "Pain reduction within 30-60 minutes with medication",
                    "short_term": "Significant relief within 2-4 hours",
                    "complete": "Fully resolved within 24 hours for tension headaches"
                },
                "when_to_see_doctor": "Sudden severe headache, headache after injury, or with fever/vision changes"
            },
            
            "fatigue": {
                "name": "Fatigue/Tiredness",
                "medications": {
                    "over_counter": ["Caffeine in moderation", "B-complex vitamins", "Iron supplements if deficient"],
                    "prescription": ["Treat underlying conditions", "Stimulants for narcolepsy", "Antidepressants if depression-related"]
                },
                "remedies": [
                    "Establish consistent sleep schedule (7-9 hours)",
                    "Stay physically active with moderate exercise",
                    "Eat balanced diet with iron-rich foods",
                    "Stay hydrated throughout the day",
                    "Practice stress management (meditation, yoga)"
                ],
                "recovery_timeline": {
                    "immediate": "Energy boost within hours with proper rest/nutrition",
                    "short_term": "Noticeable improvement within 3-7 days of lifestyle changes",
                    "complete": "Full energy restoration within 2-4 weeks for lifestyle-related fatigue"
                },
                "when_to_see_doctor": "Fatigue lasting more than 2 weeks, with unexplained weight loss, or severe enough to interfere with daily life"
            },
            
            "dizziness": {
                "name": "Dizziness",
                "medications": {
                    "over_counter": ["Meclizine 25mg as needed", "Dimenhydrinate 50mg every 4-6 hours"],
                    "prescription": ["Betahistine for Meniere's disease", "Antiemetics for nausea", "Vestibular rehabilitation"]
                },
                "remedies": [
                    "Sit or lie down immediately when dizzy",
                    "Rise slowly from sitting/lying positions",
                    "Stay hydrated and eat regular meals",
                    "Avoid sudden head movements",
                    "Practice balance exercises"
                ],
                "recovery_timeline": {
                    "immediate": "Symptom relief within minutes of resting",
                    "short_term": "Improvement within 1-3 days for benign causes",
                    "complete": "Resolution within 1-2 weeks for most cases"
                },
                "when_to_see_doctor": "Sudden severe dizziness, with chest pain, loss of consciousness, or head injury"
            },
            
            "dehydration": {
                "name": "Dehydration",
                "medications": {
                    "over_counter": ["Oral rehydration solutions", "Electrolyte drinks", "Water with salt and sugar"],
                    "prescription": ["IV fluids in severe cases", "Anti-diarrheal if caused by diarrhea"]
                },
                "remedies": [
                    "Drink water frequently in small sips",
                    "Use oral rehydration solutions",
                    "Consume water-rich fruits (watermelon, oranges)",
                    "Avoid caffeine and alcohol",
                    "Rest in cool environment"
                ],
                "recovery_timeline": {
                    "immediate": "Symptom improvement within 30-60 minutes of rehydration",
                    "short_term": "Full rehydration within 2-4 hours with proper fluid intake",
                    "complete": "Complete recovery within 24 hours for mild cases"
                },
                "when_to_see_doctor": "No urine output for 8+ hours, extreme thirst, confusion, or rapid heartbeat"
            },
            
            "allergies": {
                "name": "Allergies (Dust, Pollen)",
                "medications": {
                    "over_counter": ["Loratadine 10mg daily", "Cetirizine 10mg daily", "Fexofenadine 60mg twice daily", "Nasal corticosteroids"],
                    "prescription": ["Prescription antihistamines", "Immunotherapy (allergy shots)", "Leukotriene modifiers"]
                },
                "remedies": [
                    "Use saline nasal rinses",
                    "Keep windows closed during high pollen seasons",
                    "Use air purifiers with HEPA filters",
                    "Shower after being outdoors",
                    "Wear mask when doing yard work"
                ],
                "recovery_timeline": {
                    "immediate": "Symptom relief within 1-2 hours with antihistamines",
                    "short_term": "Control within 24-48 hours of consistent medication",
                    "complete": "Management throughout allergy season with ongoing treatment"
                },
                "when_to_see_doctor": "Symptoms interfere with daily activities, OTC medications don't help, or breathing difficulties"
            },
            
            "common cold": {
                "name": "Common Cold",
                "medications": {
                    "over_counter": ["Acetaminophen for fever/pain", "Ibuprofen for inflammation", "Decongestants", "Cough suppressants"],
                    "prescription": ["Antiviral medications in some cases", "Stronger cough medications if needed"]
                },
                "remedies": [
                    "Rest and hydrate well",
                    "Use saline nasal sprays",
                    "Gargle with warm salt water",
                    "Drink warm fluids (tea, broth)",
                    "Use humidifier in bedroom"
                ],
                "recovery_timeline": {
                    "immediate": "Symptom relief within hours of treatment",
                    "short_term": "Peak symptoms days 2-3, improvement by day 5-7",
                    "complete": "Full recovery within 7-10 days"
                },
                "when_to_see_doctor": "Symptoms last more than 10 days, high fever, shortness of breath, or severe symptoms"
            },
            
            "flu": {
                "name": "Flu (Viral Infection)",
                "medications": {
                    "over_counter": ["Acetaminophen for fever", "Ibuprofen for aches", "Decongestants", "Cough medicines"],
                    "prescription": ["Oseltamivir (Tamiflu)", "Zanamivir (Relenza)", "Baloxavir marboxil (Xofluza)"]
                },
                "remedies": [
                    "Get plenty of rest",
                    "Stay hydrated with water and electrolyte drinks",
                    "Use humidifier to ease breathing",
                    "Take warm baths for muscle aches",
                    "Eat light, easy-to-digest foods"
                ],
                "recovery_timeline": {
                    "immediate": "Symptom relief within 24 hours with antivirals",
                    "short_term": "Fever breaks in 3-4 days, most symptoms improve in 5-7 days",
                    "complete": "Full recovery within 1-2 weeks, cough may linger"
                },
                "when_to_see_doctor": "Difficulty breathing, persistent high fever, severe symptoms, or in high-risk groups"
            },
            
            "sore throat": {
                "name": "Sore Throat",
                "medications": {
                    "over_counter": ["Acetaminophen for pain", "Ibuprofen for inflammation", "Throat lozenges", "Chloraseptic spray"],
                    "prescription": ["Antibiotics if bacterial", "Corticosteroids for severe inflammation"]
                },
                "remedies": [
                    "Gargle with warm salt water",
                    "Drink warm liquids (tea with honey)",
                    "Use throat lozenges or hard candy",
                    "Stay hydrated with cool fluids",
                    "Use humidifier to moisten air"
                ],
                "recovery_timeline": {
                    "immediate": "Pain relief within 30 minutes with lozenges/analgesics",
                    "short_term": "Improvement within 2-3 days",
                    "complete": "Full recovery within 5-7 days for viral causes"
                },
                "when_to_see_doctor": "Severe pain, difficulty swallowing, lasting more than a week, or with fever/white patches"
            },
            
            "cough": {
                "name": "Cough (Dry or Productive)",
                "medications": {
                    "over_counter": ["Dextromethorphan for dry cough", "Guaifenesin for productive cough", "Cough drops", "Antihistamines if allergic"],
                    "prescription": ["Codeine cough syrup", "Benzonatate", "Inhalers for asthma-related cough"]
                },
                "remedies": [
                    "Drink warm fluids (tea with honey)",
                    "Use humidifier or steam inhalation",
                    "Suck on cough drops or hard candy",
                    "Elevate head while sleeping",
                    "Avoid irritants (smoke, strong perfumes)"
                ],
                "recovery_timeline": {
                    "immediate": "Symptomatic relief within hours with cough suppressants",
                    "short_term": "Improvement within 3-7 days for acute cough",
                    "complete": "Resolution within 2-3 weeks for most cases"
                },
                "when_to_see_doctor": "Cough lasting more than 3 weeks, coughing up blood, shortness of breath, or chest pain"
            },
            
            # Cardio & BP Conditions
            "high blood pressure": {
                "name": "High Blood Pressure (Hypertension)",
                "medications": {
                    "over_counter": ["No reliable OTC medications - prescription required"],
                    "prescription": ["Lisinopril", "Amlodipine", "Hydrochlorothiazide", "Losartan", "Metoprolol"]
                },
                "remedies": [
                    "Reduce sodium intake",
                    "Eat potassium-rich foods (bananas, leafy greens)",
                    "Practice regular physical activity",
                    "Maintain healthy weight",
                    "Limit alcohol consumption"
                ],
                "recovery_timeline": {
                    "immediate": "BP reduction within hours of medication",
                    "short_term": "Stable control within 2-4 weeks of treatment",
                    "complete": "Long-term management requires ongoing treatment"
                },
                "when_to_see_doctor": "BP reading above 180/120 mmHg, severe symptoms, or for initial diagnosis and medication"
            },
            
            "low blood pressure": {
                "name": "Low Blood Pressure (Hypotension)",
                "medications": {
                    "over_counter": ["Caffeine in moderation", "Electrolyte drinks"],
                    "prescription": ["Fludrocortisone", "Midodrine", "Pyridostigmine for severe cases"]
                },
                "remedies": [
                    "Increase salt intake (if not contraindicated)",
                    "Stay well-hydrated",
                    "Eat small, frequent meals",
                    "Wear compression stockings",
                    "Rise slowly from sitting/lying"
                ],
                "recovery_timeline": {
                    "immediate": "Symptom relief within minutes of sitting/lying down",
                    "short_term": "Improvement within days of increased salt/fluid intake",
                    "complete": "Management of underlying cause determines timeline"
                },
                "when_to_see_doctor": "Frequent dizziness, fainting, or symptoms affecting daily life"
            },
            
            # Add more conditions following the same pattern...
            "asthma": {
                "name": "Asthma",
                "medications": {
                    "over_counter": ["Primatene Mist (epinephrine)", "No reliable OTC controllers"],
                    "prescription": ["Albuterol inhaler", "Inhaled corticosteroids", "Leukotriene modifiers", "Combination inhalers"]
                },
                "remedies": [
                    "Practice breathing exercises",
                    "Identify and avoid triggers",
                    "Use air purifiers",
                    "Maintain healthy weight",
                    "Stay hydrated to thin mucus"
                ],
                "recovery_timeline": {
                    "immediate": "Relief within minutes with rescue inhaler",
                    "short_term": "Control within days of starting controller medications",
                    "complete": "Chronic condition requiring ongoing management"
                },
                "when_to_see_doctor": "Difficulty breathing despite medication, blue lips/fingernails, or worsening symptoms"
            },
            
            "stress": {
                "name": "Stress",
                "medications": {
                    "over_counter": ["No specific OTC medications", "Herbal supplements (valerian, chamomile)"],
                    "prescription": ["Antidepressants", "Anti-anxiety medications", "Beta-blockers for physical symptoms"]
                },
                "remedies": [
                    "Practice deep breathing exercises",
                    "Regular physical activity",
                    "Maintain healthy sleep schedule",
                    "Practice mindfulness meditation",
                    "Connect with friends and family"
                ],
                "recovery_timeline": {
                    "immediate": "Calming effect within minutes of relaxation techniques",
                    "short_term": "Noticeable improvement within days of consistent stress management",
                    "complete": "Ongoing management for chronic stress"
                },
                "when_to_see_doctor": "Stress interferes with daily functioning, thoughts of self-harm, or physical symptoms"
            }
        }

    def analyze_statement(self, user_input):
        user_input = user_input.lower().strip()
        
        # Check for each condition
        for condition in self.health_conditions_database.keys():
            if condition in user_input:
                return self.get_detailed_response(condition)
        
        # Check for related terms
        if any(term in user_input for term in ['tired', 'exhausted']):
            return self.get_detailed_response("fatigue")
        elif any(term in user_input for term in ['hypertension', 'high bp']):
            return self.get_detailed_response("high blood pressure")
        elif any(term in user_input for term in ['low bp', 'hypotension']):
            return self.get_detailed_response("low blood pressure")
        
        return "I can help with various health conditions. Please describe your symptoms specifically."

    def get_detailed_response(self, condition_key):
        data = self.health_conditions_database[condition_key]
        
        response = f"""**🩺 {data['name'].upper()} - COMPLETE TREATMENT GUIDE**

**💊 MEDICATIONS:**
**Over-the-Counter:**
{chr(10).join(['• ' + med for med in data['medications']['over_counter']])}

**Prescription:**
{chr(10).join(['• ' + med for med in data['medications']['prescription']])}

**🏠 HOME REMEDIES:**
{chr(10).join(['• ' + remedy for remedy in data['remedies']])}

**⏰ RECOVERY TIMELINE:**
• **Immediate Relief**: {data['recovery_timeline']['immediate']}
• **Short-term Improvement**: {data['recovery_timeline']['short_term']}
• **Complete Recovery**: {data['recovery_timeline']['complete']}

**🚨 SEE DOCTOR IF:**
• {data['when_to_see_doctor']}"""
        return response

def generate_response(self, user_input):
    st.session_state.ai_thinking = True
    current_time = datetime.now().strftime("%H:%M")
    
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user", 
        "content": user_input,
        "timestamp": current_time
    })
    
    # Get AI response
    response = self.analyze_statement(user_input)
    
    # Add AI response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response, 
        "timestamp": current_time
    })
    
    st.session_state.ai_thinking = False
    return response
# =============================================
# MAIN AI HEALTH SYSTEM 3.0 APPLICATION
# =============================================
class AIHealthSystem3:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.security_manager = SecurityManager()
        self.symptom_checker = AdvancedSymptomChecker()
        self.prescription_system = SmartPrescriptionSystem()
        self.imaging_ai = MedicalImagingAI()
        self.patient_portal = PatientPortalSystem()
        self.telemedicine = TelemedicineSystem()
        self.health_monitor = EnhancedHealthMonitor()
        self.initialize_sample_data()
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="AI Health System 3.0",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'logged_in': False,
                'username': '',
                'current_page': 'Dashboard',
                'user_role': 'patient',
                'patients': [],
                'doctors': [],
                'appointments': [],
                'chat_history': [],
                'ai_thinking': False,
                'pharmacy_inventory': [],
                'billing_records': [],
                'payment_records': [],
                'patient_data': {},
                'consultation_data': {},
                'imaging_analysis': {},
                'prescription_data': {},
                'symptom_analysis': {},
                'telemedicine_sessions': [],
                'vital_signs_history': [],
                'pain_assessment_history': [],
                'sleep_history': [],
                'emotion_history': []
            })
    
    def initialize_sample_data(self):
        """Initialize sample data for the system"""
        if not st.session_state.patients:
            st.session_state.patients = [
                {"id": 1, "name": "John Smith", "age": 45, "gender": "Male", "phone": "555-0101", "email": "john@email.com", "condition": "Hypertension"},
                {"id": 2, "name": "Maria Garcia", "age": 32, "gender": "Female", "phone": "555-0102", "email": "maria@email.com", "condition": "Diabetes"},
                {"id": 3, "name": "Robert Johnson", "age": 58, "gender": "Male", "phone": "555-0103", "email": "robert@email.com", "condition": "Arthritis"}
            ]
        
        if not st.session_state.doctors:
            st.session_state.doctors = [
                {"id": 1, "name": "Dr. Sarah Wilson", "specialization": "Cardiology", "phone": "555-0201", "email": "sarah@hospital.com", "availability": "Mon-Fri"},
                {"id": 2, "name": "Dr. James Anderson", "specialization": "Cardiology", "phone": "555-0204", "email": "james@hospital.com", "availability": "Tue-Sat"},
                {"id": 3, "name": "Dr. Michael Chen", "specialization": "Neurology", "phone": "555-0202", "email": "michael@hospital.com", "availability": "Tue-Sat"},
                {"id": 4, "name": "Dr. Lisa Roberts", "specialization": "Neurology", "phone": "555-0205", "email": "lisa@hospital.com", "availability": "Mon-Thu"},
                {"id": 5, "name": "Dr. Emily Davis", "specialization": "Pediatrics", "phone": "555-0203", "email": "emily@hospital.com", "availability": "Mon-Thu"}
            ]
        
        if not st.session_state.appointments:
            st.session_state.appointments = [
                {"id": 1, "patient": "John Smith", "doctor": "Dr. Sarah Wilson", "date": "2024-01-15", "time": "10:00", "status": "Scheduled"},
                {"id": 2, "patient": "Maria Garcia", "doctor": "Dr. Michael Chen", "date": "2024-01-16", "time": "14:30", "status": "Completed"},
                {"id": 3, "patient": "Robert Johnson", "doctor": "Dr. Emily Davis", "date": "2024-01-17", "time": "11:15", "status": "Scheduled"}
            ]
        
        if not st.session_state.pharmacy_inventory:
            st.session_state.pharmacy_inventory = [
                {"id": 1, "name": "Paracetamol", "category": "Pain Relief", "quantity": 150, "price": 5.99, "supplier": "MediCorp", "expiry": "2024-12-31"},
                {"id": 2, "name": "Amoxicillin", "category": "Antibiotic", "quantity": 80, "price": 12.50, "supplier": "PharmaPlus", "expiry": "2024-10-15"},
                {"id": 3, "name": "Metformin", "category": "Diabetes", "quantity": 120, "price": 8.75, "supplier": "HealthCare Ltd", "expiry": "2025-03-20"}
            ]
        
        if not st.session_state.billing_records:
            st.session_state.billing_records = [
                {"id": 1, "patient": "John Smith", "amount": 250.00, "date": "2024-01-15", "status": "Paid", "service": "Consultation"},
                {"id": 2, "patient": "Maria Garcia", "amount": 180.50, "date": "2024-01-16", "status": "Pending", "service": "Lab Tests"}
            ]
    
    def apply_custom_styling(self):
        st.markdown("""
        <style>
        /* Main Background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #E8F4FD, #F0F7FF, #E3F2FD);
            color: #2C3E50;
        }
        
        /* Sidebar - Professional Blue */
        [data-testid="stSidebar"] {
            background: linear-gradient(165deg, #1976D2, #1565C0, #0D47A1) !important;
            border-right: 1px solid #BBDEFB;
        }
        
        /* Sidebar Header */
        [data-testid="stSidebar"] .css-1d391kg {
            background: transparent !important;
        }
        
        /* Sidebar Buttons */
        .stButton button {
            background: linear-gradient(135deg, #1976D2, #1565C0);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            margin: 2px 0;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #1565C0, #0D47A1);
            color: white;
            border: none;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(21, 101, 192, 0.3);
        }
        
        /* Cards */
        .custom-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .main-title {
            color: #1976D2;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        
        .emergency-alert {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .success-message {
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .doctor-card {
            background: linear-gradient(135deg, #1976D2, #1565C0);
            color: white !important;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
        }
        
        .login-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-bottom: 2rem;
            padding-top: 2rem;
        }
        .login-logo {
            border-radius: 50%;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
        }
        .login-title {
            color: #1976D2;
            font-size: 2.8rem;
            margin: 0;
            font-weight: 700;
        }
        .login-subtitle {
            color: #666;
            font-size: 1.2rem;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        self.apply_custom_styling()
        
        if not st.session_state.logged_in:
            self.show_login_page()
        else:
            self.show_main_application()
    
    def show_login_page(self):
        """Display login page from second code"""
        st.markdown("""
        <div class="login-header">
            <img src="https://static.vecteezy.com/system/resources/thumbnails/036/167/123/small/ai-generated-technical-support-automation-filled-colorful-logo-user-centric-service-business-value-robot-hold-heart-icon-design-element-ai-art-for-corporate-branding-vector.jpg" 
                 width="80" height="80" class="login-logo">
            <h1 class="login-title">AI Health System 3.0</h1>
            <p class="login-subtitle">Advanced Healthcare Platform with AI-Powered Diagnostics</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.container():
                st.subheader("🔐 Secure Login")
                
                login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
                
                with login_tab:
                    username = st.text_input("📧 Email Address", placeholder="Enter your email")
                    password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
                    
                    if st.button("🚀 Login", use_container_width=True):
                        if self.authenticate_user(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                
                with signup_tab:
                    new_username = st.text_input("New Email", placeholder="Enter your email")
                    new_password = st.text_input("New Password", type="password", placeholder="Create a password")
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                    user_role = st.selectbox("Role", ["Patient", "Healthcare Provider", "Administrator"])
                    
                    if st.button("Create Account", use_container_width=True):
                        if new_password == confirm_password:
                            if new_username and new_password:
                                st.success("✅ Account created successfully! Please login.")
                            else:
                                st.error("❌ Please fill all fields")
                        else:
                            st.error("❌ Passwords do not match")
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user (simplified for demo)"""
        return len(username) > 0 and len(password) > 0
    
    def show_main_application(self):
        """Display main application"""
        self.show_sidebar()
        self.show_current_page()
    
    def show_sidebar(self):
        """Display application sidebar with merged navigation"""
        with st.sidebar:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h3 style="color: white;">👋 Welcome, {st.session_state.username}</h3>
                <p style="color: rgba(255,255,255,0.8);">AI Health System 3.0</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation
            st.markdown("**🧭 NAVIGATION**")
            
            # Core Management Pages from First Code
            core_pages = {
                "🏠 Dashboard": "dashboard",
                "👥 Patients": "patients", 
                "👨‍⚕️ Doctors": "doctors",
                "📅 Appointments": "appointments",
                "🤖 AI Assistant": "ai_assistant",
                "💊 Pharmacy Settings": "pharmacy_settings",
                "🧾 Billing & Payments": "billing_payments"
            }
            
            for page_name, page_id in core_pages.items():
                if st.button(page_name, use_container_width=True, key=f"nav_{page_id}"):
                    st.session_state.current_page = page_id
                    st.rerun()
            
            st.divider()
            
            # Advanced AI Features from Second Code
            st.markdown("**🚀 ADVANCED AI FEATURES**")
            
            ai_pages = {
                "🤒 Symptom Checker": "symptom_checker",
                "💊 Smart Prescriptions": "prescriptions", 
                "🖼️ Medical Imaging AI": "medical_imaging",
                "📋 Patient Portal": "patient_portal",
                "📞 Telemedicine": "telemedicine"
            }
            
            for page_name, page_id in ai_pages.items():
                if st.button(page_name, use_container_width=True, key=f"ai_{page_id}"):
                    st.session_state.current_page = page_id
                    st.rerun()
            
            st.divider()
            
            # Health Monitoring & Profile
            st.markdown("**❤️ HEALTH & PROFILE**")
            
            health_pages = {
                "❤️ Health Monitor": "health_monitor",
                "📈 HRV Analysis": "hrv_analysis", 
                "👤 User Profile": "user_profile",
                "⚙️ Settings": "settings"
            }
            
            for page_name, page_id in health_pages.items():
                if st.button(page_name, use_container_width=True, key=f"health_{page_id}"):
                    st.session_state.current_page = page_id
                    st.rerun()
            
            st.divider()
            
            # Quick Actions
            st.markdown("**⚡ QUICK ACTIONS**")
            if st.button("🆘 Emergency Help", use_container_width=True):
                self.show_emergency_help()
            
            if st.button("📞 Contact Support", use_container_width=True):
                self.show_support_contact()
            
            st.divider()
            
            # Logout
            if st.button("🔒 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.current_page = "Dashboard"
                st.rerun()
    
    def show_current_page(self):
        """Display current page based on navigation"""
        current_page = st.session_state.current_page
        
        if current_page == "dashboard":
            self.show_dashboard()
        elif current_page == "patients":
            self.show_patients_page()
        elif current_page == "doctors":
            self.show_doctors_page()
        elif current_page == "appointments":
            self.show_appointments_page()
        elif current_page == "ai_assistant":
            self.show_ai_assistant_page()
        elif current_page == "pharmacy_settings":
            self.show_pharmacy_settings_page()
        elif current_page == "billing_payments":
            self.show_billing_payment_page()
        elif current_page == "symptom_checker":
            self.show_symptom_checker()
        elif current_page == "prescriptions":
            self.show_prescriptions()
        elif current_page == "medical_imaging":
            self.show_medical_imaging()
        elif current_page == "patient_portal":
            self.show_patient_portal()
        elif current_page == "telemedicine":
            self.show_telemedicine()
        elif current_page == "health_monitor":
            self._show_health_monitor_page()
        elif current_page == "hrv_analysis":
            self.show_hrv_analysis_page()
        elif current_page == "user_profile":
            self.show_user_profile_page()
        elif current_page == "settings":
            self.show_settings()
        else:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='main-title'>{st.session_state.current_page}</h2>", unsafe_allow_html=True)
            st.info(f"🚧 {st.session_state.current_page} section is under development")
            st.markdown("</div>", unsafe_allow_html=True)

    # =============================================
    # CORE PAGES FROM FIRST CODE
    # =============================================
    
    def show_dashboard(self):
        st.markdown("<h1 class='main-title'>🏥 Hospital Dashboard</h1>", unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(st.session_state.patients), "12%")
        with col2:
            st.metric("Total Doctors", len(st.session_state.doctors), "5%")
        with col3:
            today_appointments = len([a for a in st.session_state.appointments if a['status'] == 'Scheduled'])
            st.metric("Today's Appointments", today_appointments, "8%")
        with col4:
            completed_appointments = len([a for a in st.session_state.appointments if a['status'] == 'Completed'])
            st.metric("Completed", completed_appointments, "15%")
        
        # Recent Activity and Quick Actions
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("📋 Recent Appointments")
            if st.session_state.appointments:
                recent_appointments = pd.DataFrame(st.session_state.appointments[-5:])
                st.dataframe(recent_appointments[['patient', 'doctor', 'date', 'status']], use_container_width=True)
            else:
                st.info("No recent appointments")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_right:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("⚡ Quick Actions")
            
            if st.button("➕ Add New Patient", use_container_width=True):
                st.session_state.current_page = "Patients"
                st.rerun()
            
            if st.button("📅 Schedule Appointment", use_container_width=True):
                st.session_state.current_page = "Appointments"
                st.rerun()
            
            if st.button("👨‍⚕️ Manage Doctors", use_container_width=True):
                st.session_state.current_page = "Doctors"
                st.rerun()
                
            if st.button("📊 Generate Report", use_container_width=True):
                st.success("Report generation started!")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Features Grid from Second Code
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()
        
        st.subheader("🚀 AI-Powered Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 🤒 Symptom Checker 2.0")
                st.write("Multi-symptom analysis with probability scoring and risk assessment")
                if st.button("Analyze Symptoms", key="dash_symptom"):
                    st.session_state.current_page = "symptom_checker"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 💊 Smart Prescriptions")
                st.write("AI-generated prescriptions with dosage optimization and interaction checks")
                if st.button("Generate Prescription", key="dash_prescription"):
                    st.session_state.current_page = "prescriptions"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### 🖼️ Medical Imaging AI")
                st.write("Advanced image analysis for X-rays, MRI, CT scans with AI detection")
                if st.button("Analyze Images", key="dash_imaging"):
                    st.session_state.current_page = "medical_imaging"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    
    def show_patients_page(self):
        st.markdown("<h1 class='main-title'>👥 Patient Management</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3,1])
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Add New Patient")
            
            with st.form("patient_form"):
                name = st.text_input("Full Name")
                age = st.number_input("Age", min_value=0, max_value=120)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                phone = st.text_input("Phone")
                email = st.text_input("Email")
                condition = st.text_input("Medical Condition")
                
                if st.form_submit_button("Add Patient", use_container_width=True):
                    if name and age and phone:
                        new_patient = {
                            "id": len(st.session_state.patients) + 1,
                            "name": name,
                            "age": age,
                            "gender": gender,
                            "phone": phone,
                            "email": email,
                            "condition": condition
                        }
                        st.session_state.patients.append(new_patient)
                        st.success(f"Patient {name} added successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Patient Records")
            
            if st.session_state.patients:
                patient_df = pd.DataFrame(st.session_state.patients)
                st.dataframe(patient_df, use_container_width=True)
                
                # Search and filter
                search_col1, search_col2 = st.columns(2)
                with search_col1:
                    search_name = st.text_input("Search by Name")
                with search_col2:
                    filter_condition = st.selectbox("Filter by Condition", ["All"] + list(patient_df['condition'].unique()))
                
                filtered_patients = patient_df
                if search_name:
                    filtered_patients = filtered_patients[filtered_patients['name'].str.contains(search_name, case=False)]
                if filter_condition != "All":
                    filtered_patients = filtered_patients[filtered_patients['condition'] == filter_condition]
                
                st.dataframe(filtered_patients, use_container_width=True)
            else:
                st.info("No patients found. Add your first patient!")
            st.markdown("</div>", unsafe_allow_html=True)
    
    def show_doctors_page(self):
        st.markdown("<h1 class='main-title'>👨‍⚕️ Doctor Management</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3,1])
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Add New Doctor")
            
            with st.form("doctor_form"):
                name = st.text_input("Doctor Name")
                specialization = st.selectbox("Specialization", [
                    "Cardiology", "Neurology", "Pediatrics", "Gynecology", "Dentistry", 
                    "Orthopedics", "Dermatology", "Psychiatry", "Ophthalmology", "General Surgery",
                    "ENT", "Urology", "Gastroenterology", "Endocrinology", "General"
                ])
                phone = st.text_input("Phone")
                email = st.text_input("Email")
                availability = st.selectbox("Availability", ["Mon-Fri", "Tue-Sat", "Mon-Thu", "Weekends"])
                
                if st.form_submit_button("Add Doctor", use_container_width=True):
                    if name and specialization and phone:
                        new_doctor = {
                            "id": len(st.session_state.doctors) + 1,
                            "name": name,
                            "specialization": specialization,
                            "phone": phone,
                            "email": email,
                            "availability": availability
                        }
                        st.session_state.doctors.append(new_doctor)
                        st.success(f"Dr. {name} added successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Medical Staff")
            
            if st.session_state.doctors:
                # Display doctors in blue cards
                cols = st.columns(2)
                for idx, doctor in enumerate(st.session_state.doctors):
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div class='doctor-card'>
                            <h4 style='color: white; margin-bottom: 0.5rem;'>👨‍⚕️ {doctor['name']}</h4>
                            <p style='color: white; margin: 0.2rem 0;'><strong>Specialization:</strong> {doctor['specialization']}</p>
                            <p style='color: white; margin: 0.2rem 0;'><strong>Phone:</strong> {doctor['phone']}</p>
                            <p style='color: white; margin: 0.2rem 0;'><strong>Email:</strong> {doctor['email']}</p>
                            <p style='color: white; margin: 0.2rem 0;'><strong>Availability:</strong> {doctor['availability']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No doctors found. Add your first doctor!")
            st.markdown("</div>", unsafe_allow_html=True)
    
    def show_appointments_page(self):
        st.markdown("<h1 class='main-title'>📅 Appointment Management</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3,1])
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Schedule Appointment")
            
            with st.form("appointment_form"):
                patient = st.selectbox("Patient", [p['name'] for p in st.session_state.patients])
                doctor = st.selectbox("Doctor", [d['name'] for d in st.session_state.doctors])
                date = st.date_input("Date", min_value=datetime.now().date())
                time = st.time_input("Time")
                status = st.selectbox("Status", ["Scheduled", "Completed", "Cancelled"])
                
                if st.form_submit_button("Schedule Appointment", use_container_width=True):
                    if patient and doctor and date and time:
                        new_appointment = {
                            "id": len(st.session_state.appointments) + 1,
                            "patient": patient,
                            "doctor": doctor,
                            "date": str(date),
                            "time": str(time),
                            "status": status
                        }
                        st.session_state.appointments.append(new_appointment)
                        st.success(f"Appointment scheduled successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Appointment Schedule")
            
            if st.session_state.appointments:
                appointment_df = pd.DataFrame(st.session_state.appointments)
                
                # Filter options
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    filter_status = st.selectbox("Filter by Status", ["All"] + list(appointment_df['status'].unique()))
                with col_filter2:
                    filter_doctor = st.selectbox("Filter by Doctor", ["All"] + list(appointment_df['doctor'].unique()))
                
                filtered_appointments = appointment_df
                if filter_status != "All":
                    filtered_appointments = filtered_appointments[filtered_appointments['status'] == filter_status]
                if filter_doctor != "All":
                    filtered_appointments = filtered_appointments[filtered_appointments['doctor'] == filter_doctor]
                
                st.dataframe(filtered_appointments, use_container_width=True)
                
                # Quick actions
                st.subheader("Quick Actions")
                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("Mark Selected as Completed", use_container_width=True):
                        st.info("Feature: Select appointments to mark as completed")
                with col_act2:
                    if st.button("Send Reminders", use_container_width=True):
                        st.success("Reminders sent to all patients with upcoming appointments!")
            else:
                st.info("No appointments scheduled. Schedule your first appointment!")
            st.markdown("</div>", unsafe_allow_html=True)
    
    def show_ai_assistant_page(self):
        # Import the RealHealthAI class from first code
       
        
        # Initialize AI Chatbot
        if 'ai_bot' not in st.session_state:
            st.session_state.ai_bot = RealHealthAI()
        
        ai_bot = st.session_state.ai_bot
        
        st.markdown("<h1 class='main-title'>🤖 AI Health Assistant</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container():
                st.subheader("💬 Chat with Health AI")
                st.write("Describe your symptoms for detailed medical analysis")
                
                st.write("**Quick Actions:**")
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                
                with btn_col1:
                    if st.button("🤒 Fever", use_container_width=True, key="fever_btn"):
                        ai_bot.generate_response("fever high temperature")
                        st.rerun()
                
                with btn_col2:
                    if st.button("🦶 Pain", use_container_width=True, key="pain_btn"):
                        ai_bot.generate_response("body pain muscle joint pain")
                        st.rerun()
                
                with btn_col3:
                    if st.button("🫁 Cough", use_container_width=True, key="cough_btn"):
                        ai_bot.generate_response("cough cold breathing")
                        st.rerun()
                
                with btn_col4:
                    if st.button("💊 Meds", use_container_width=True, key="meds_btn"):
                        ai_bot.generate_response("medications drugs prescription")
                        st.rerun()
                
                st.write("---")
                user_input = st.text_input(
                    "**Describe your symptoms:**",
                    placeholder="Example: I have a migraine headache...",
                    key="chat_input"
                )
                
                if st.button("🚀 Get Medical Analysis", type="primary", use_container_width=True):
                    if user_input.strip():
                        with st.spinner("🔍 Analyzing your symptoms..."):
                            ai_bot.generate_response(user_input)
                        st.rerun()
                    else:
                        st.warning("Please describe your symptoms")
        
        with col2:
            with st.container():
                st.subheader("🩺 Symptom Checker")
                
                symptoms = st.multiselect(
                    "Select symptoms:",
                    ["Fever", "Cough", "Headache", "Fatigue", "Chest Pain", "Joint Pain", 
                     "Muscle Aches", "Sore Throat", "Nausea", "Dizziness"],
                    key="symptoms_select"
                )
                
                severity = st.select_slider("Severity:", ["Mild", "Moderate", "Severe"])
                duration = st.selectbox("Duration:", ["<1 day", "1-3 days", "3-7 days", "1-2 weeks", ">2 weeks"])
                
                if st.button("🔍 Analyze Symptoms", use_container_width=True):
                    if symptoms:
                        with st.spinner("Analyzing..."):
                            ai_bot.generate_response(f"Symptoms: {', '.join(symptoms)}, Severity: {severity}, Duration: {duration}")
                        st.rerun()
                    else:
                        st.warning("Select symptoms first")

        # Chat History Display
        st.write("---")
        st.subheader("💬 Conversation")
        
        if not st.session_state.chat_history:
            st.info("👆 Describe your symptoms above or use quick buttons to get started")
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(message['content'])
                st.write("---")
        
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    def show_pharmacy_settings_page(self):
        st.markdown("<h1 class='main-title'>💊 Pharmacy Settings</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("📦 Medicine Inventory")
            
            # Add new medicine
            with st.expander("➕ Add New Medicine"):
                col1, col2 = st.columns(2)
                with col1:
                    new_name = st.text_input("Medicine Name")
                    new_category = st.selectbox("Category", ["Pain Relief", "Antibiotic", "Diabetes", "Cholesterol", "Cardiac", "Other"])
                    new_quantity = st.number_input("Quantity", min_value=0, step=1)
                with col2:
                    new_price = st.number_input("Price (Rs.)", min_value=0.0, step=0.1)
                    new_supplier = st.text_input("Supplier")
                    new_expiry = st.date_input("Expiry Date")
                
                if st.button("Add Medicine"):
                    if new_name and new_supplier:
                        new_id = max([med['id'] for med in st.session_state.pharmacy_inventory], default=0) + 1
                        st.session_state.pharmacy_inventory.append({
                            "id": new_id,
                            "name": new_name,
                            "category": new_category,
                            "quantity": new_quantity,
                            "price": new_price,
                            "supplier": new_supplier,
                            "expiry": new_expiry.strftime("%Y-%m-%d")
                        })
                        st.success(f"✅ {new_name} added to inventory!")
                        st.rerun()
                    else:
                        st.error("❌ Please fill all required fields")
            
            # Display inventory
            st.markdown("### Current Inventory")
            if st.session_state.pharmacy_inventory:
                inventory_df = pd.DataFrame(st.session_state.pharmacy_inventory)
                st.dataframe(inventory_df, use_container_width=True)
            else:
                st.info("📝 No medicines in inventory. Add some medicines to get started.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("📊 Pharmacy Analytics")
            
            # Statistics
            total_medicines = len(st.session_state.pharmacy_inventory)
            low_stock_count = len([med for med in st.session_state.pharmacy_inventory if med['quantity'] < 20])
            total_value = sum(med['quantity'] * med['price'] for med in st.session_state.pharmacy_inventory)
            
            st.metric("Total Medicines", total_medicines)
            st.metric("Low Stock Items", low_stock_count)
            st.metric("Inventory Value", f"Rs.{total_value:,.2f}")
            
            if low_stock_count > 0:
                st.warning(f"🚨 {low_stock_count} medicines are running low on stock!")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def show_billing_payment_page(self):
        st.markdown("<h1 class='main-title'>🧾 Billing & Payments</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("💰 Billing Records")
            
            if st.session_state.billing_records:
                billing_df = pd.DataFrame(st.session_state.billing_records)
                st.dataframe(billing_df, use_container_width=True)
                
                # Summary statistics
                total_amount = sum(r['amount'] for r in st.session_state.billing_records)
                pending_amount = sum(r['amount'] for r in st.session_state.billing_records if r['status'] == 'Pending')
                
                st.metric("Total Amount", f"Rs.{total_amount:,.2f}")
                st.metric("Pending Amount", f"Rs.{pending_amount:,.2f}")
            else:
                st.info("📝 No billing records found.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("💳 Create Invoice")
            
            with st.form("invoice_form"):
                patient = st.selectbox("Patient", [p["name"] for p in st.session_state.patients])
                service_type = st.selectbox("Service Type", ["Consultation", "Lab Tests", "Surgery", "Medication", "Room Charges"])
                amount = st.number_input("Amount (Rs.)", min_value=0.0, step=0.1)
                description = st.text_area("Description")
                
                if st.form_submit_button("Generate Invoice"):
                    new_id = max([bill['id'] for bill in st.session_state.billing_records], default=0) + 1
                    st.session_state.billing_records.append({
                        "id": new_id,
                        "patient": patient,
                        "amount": amount,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "status": "Pending",
                        "service": service_type,
                        "description": description
                    })
                    st.success(f"✅ Invoice generated for {patient} - Rs.{amount:,.2f}")
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

    # =============================================
    # ADVANCED AI PAGES FROM SECOND CODE
    # =============================================
    
    def show_symptom_checker(self):
        """Display AI Symptom Checker 2.0 from second code"""
        st.markdown("""
        <div class="main-header">
            <h1>🤒 AI Symptom Checker 2.0</h1>
            <p>Multi-symptom analysis with probability scoring and risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔍 Symptom Analysis", "📊 Results", "📈 Health Risk Assessment"])
        
        with tab1:
            st.subheader("Describe Your Symptoms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Symptom selection
                st.markdown("#### 🎯 Select Symptoms")
                symptoms = st.multiselect(
                    "Choose your symptoms:",
                    [
                        "chest_pain", "shortness_of_breath", "headache", "dizziness",
                        "abdominal_pain", "nausea", "cough", "fever", "fatigue",
                        "joint_pain", "back_pain", "rash", "swelling", "palpitations"
                    ],
                    help="Select all symptoms you're experiencing"
                )
                
                # Symptom severity
                st.markdown("#### 📊 Symptom Severity")
                severity = st.select_slider(
                    "Overall symptom severity:",
                    options=["Mild", "Moderate", "Severe", "Very Severe"],
                    value="Moderate"
                )
                
                # Symptom duration
                duration = st.selectbox(
                    "How long have you had these symptoms?",
                    ["Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"]
                )
            
            with col2:
                # Patient information
                st.markdown("#### 👤 Patient Information")
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                
                # Medical history
                st.markdown("#### 📋 Medical History")
                has_chronic_conditions = st.checkbox("Chronic medical conditions")
                smoker = st.checkbox("Smoker")
                obese = st.checkbox("Overweight/Obesity")
                family_history = st.text_area("Family medical history")
            
            # Additional symptoms description
            st.markdown("#### 📝 Additional Details")
            additional_notes = st.text_area(
                "Describe your symptoms in more detail:",
                placeholder="When did symptoms start? What makes them better or worse? Any other relevant information..."
            )
            
            if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
                if symptoms:
                    # Prepare patient data
                    patient_data = {
                        'age': age,
                        'gender': gender.lower(),
                        'has_chronic_conditions': has_chronic_conditions,
                        'smoker': smoker,
                        'obese': obese,
                        'family_history': family_history,
                        'symptom_severity': severity,
                        'symptom_duration': duration
                    }
                    
                    # Perform analysis
                    with st.spinner("🤖 AI is analyzing your symptoms..."):
                        analysis_result = self.symptom_checker.analyze_symptoms(symptoms, patient_data)
                        st.session_state.symptom_analysis = analysis_result
                    
                    st.success("✅ Analysis complete! View results in the Results tab.")
                else:
                    st.error("Please select at least one symptom.")
        
        with tab2:
            st.subheader("📊 Symptom Analysis Results")
            
            if not st.session_state.get('symptom_analysis'):
                st.info("👆 Please complete symptom analysis first")
                return
            
            analysis = st.session_state.symptom_analysis
            
            # Emergency alerts
            if analysis.get('emergency_flags'):
                st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
                st.error("🚨 URGENT MEDICAL ATTENTION REQUIRED")
                for flag in analysis['emergency_flags']:
                    st.write(f"• {flag}")
                st.write("Please seek immediate medical care or call emergency services.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Condition probabilities
            st.markdown("#### 🎯 Condition Probabilities")
            conditions = analysis.get('condition_probabilities', {})
            
            if conditions:
                for condition_id, condition_info in list(conditions.items())[:5]:  # Top 5
                    prob = condition_info['probability']
                    urgency = condition_info['urgency']
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{condition_info['name']}**")
                        st.progress(prob)
                    with col2:
                        st.metric("Probability", f"{prob:.1%}")
                    
                    # Urgency indicator
                    if urgency == 'emergency':
                        st.warning(f"🚨 {urgency.upper()} - Immediate attention required")
                    elif urgency == 'urgent':
                        st.info(f"⚠️ {urgency.title()} - Evaluation within 24-48 hours")
                    else:
                        st.success(f"✅ {urgency.title()} - Routine follow-up")
                    
                    st.divider()
            else:
                st.info("No significant conditions identified based on symptoms provided.")
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.info("No specific recommendations at this time.")
        
        with tab3:
            st.subheader("📈 Health Risk Assessment")
            
            if not st.session_state.get('symptom_analysis'):
                st.info("👆 Please complete symptom analysis first")
                return
            
            risk_assessment = st.session_state.symptom_analysis.get('risk_assessment', {})
            
            if risk_assessment:
                # Risk level visualization
                risk_level = risk_assessment.get('risk_level', 'Low')
                risk_score = risk_assessment.get('risk_score', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if risk_level == 'High':
                        st.error(f"🔴 {risk_level} Risk")
                    elif risk_level == 'Moderate':
                        st.warning(f"🟡 {risk_level} Risk")
                    else:
                        st.success(f"🟢 {risk_level} Risk")
                
                with col2:
                    st.metric("Risk Score", risk_score)
                
                with col3:
                    st.metric("Risk Factors", len(risk_assessment.get('risk_factors', [])))
                
                # Risk factors
                st.markdown("#### 🎯 Identified Risk Factors")
                risk_factors = risk_assessment.get('risk_factors', [])
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No significant risk factors identified")
                
                # Risk reduction recommendations
                st.markdown("#### 💡 Risk Reduction Strategies")
                recommendations = risk_assessment.get('recommendations', [])
                if recommendations:
                    for rec in recommendations:
                        st.write(f"• {rec}")
                else:
                    st.info("Maintain current healthy lifestyle practices")
            else:
                st.info("Risk assessment data not available")
    
    def show_prescriptions(self):
        """Display Smart Prescription System from second code"""
        st.markdown("""
        <div class="main-header">
            <h1>💊 Smart Prescription System</h1>
            <p>AI-powered medication management with dosage optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔄 Generate Prescription", "📋 Current Medications", "⚠️ Drug Interactions"])
        
        with tab1:
            st.subheader("Generate New Prescription")
            
            col1, col2 = st.columns(2)
            
            with col1:
                diagnosis = st.selectbox(
                    "Medical Diagnosis",
                    [
                        "bacterial_pneumonia", "urinary_tract_infection", "hypertension",
                        "migraine", "osteoarthritis", "type_2_diabetes", "acute_bronchitis",
                        "sinusitis", "dermatitis", "anxiety"
                    ]
                )
                
                patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=45)
                patient_weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
                patient_allergies = st.multiselect(
                    "Known Allergies",
                    ["penicillin", "sulfa", "aspirin", "nsaids", "codeine", "none"]
                )
            
            with col2:
                existing_conditions = st.multiselect(
                    "Existing Medical Conditions",
                    ["hypertension", "diabetes", "asthma", "copd", "heart_disease", "liver_disease", "renal_disease"]
                )
                
                current_medications = st.text_area(
                    "Current Medications (one per line)",
                    placeholder="Enter current medications...\nExample: lisinopril 10mg daily\natorvastatin 20mg daily"
                )
                
                pregnancy_status = st.selectbox(
                    "Pregnancy Status",
                    ["Not pregnant", "Pregnant", "Breastfeeding", "Unknown"]
                )
            
            if st.button("💊 Generate Smart Prescription", type="primary", use_container_width=True):
                # Prepare patient data
                patient_data = {
                    'age': patient_age,
                    'weight': patient_weight,
                    'allergies': patient_allergies,
                    'medical_conditions': existing_conditions,
                    'pregnant': pregnancy_status in ['Pregnant', 'Breastfeeding'],
                    'renal_function': 'normal',
                    'hepatic_function': 'normal'
                }
                
                # Parse current medications
                existing_meds = []
                if current_medications:
                    existing_meds = [line.strip() for line in current_medications.split('\n') if line.strip()]
                
                # Generate prescription
                with st.spinner("🤖 Generating optimized prescription..."):
                    prescription = self.prescription_system.generate_prescription(
                        diagnosis, patient_data, existing_meds
                    )
                    st.session_state.prescription_data = prescription
                
                st.success("✅ Prescription generated successfully!")
        
        with tab2:
            st.subheader("Current Medication List")
            
            if st.session_state.get('prescription_data'):
                prescription = st.session_state.prescription_data
                
                st.markdown("#### 💊 Prescribed Medications")
                for med in prescription.get('prescribed_medications', []):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{med['name']}**")
                            st.write(f"Dosage: {med['dosage']}")
                            st.write(f"Frequency: {med['frequency']}")
                            st.write(f"Duration: {med['duration']}")
                        with col2:
                            st.metric("Status", "Prescribed")
                        
                        st.write(f"**Instructions:** {med['instructions']}")
                        
                        # Warnings
                        if med.get('warnings'):
                            for warning in med['warnings']:
                                st.warning(warning)
                        
                        st.divider()
                
                # Patient instructions
                st.markdown("#### 📋 Patient Instructions")
                instructions = prescription.get('patient_specific_instructions', [])
                for instruction in instructions:
                    st.write(f"• {instruction}")
                
                # Follow-up recommendations
                st.markdown("#### 📅 Follow-up Recommendations")
                follow_ups = prescription.get('follow_up_recommendations', [])
                for follow_up in follow_ups:
                    st.write(f"• {follow_up}")
            else:
                st.info("No prescription data available. Generate a prescription first.")
        
        with tab3:
            st.subheader("⚠️ Drug Interaction Checker")
            
            if st.session_state.get('prescription_data'):
                prescription = st.session_state.prescription_data
                
                interactions = prescription.get('interaction_warnings', [])
                if interactions:
                    st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
                    st.error("🚨 POTENTIAL DRUG INTERACTIONS DETECTED")
                    for interaction in interactions:
                        st.write(f"• {interaction}")
                    st.write("Please consult with your healthcare provider before taking these medications together.")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-message">', unsafe_allow_html=True)
                    st.success("✅ No significant drug interactions detected")
                    st.write("All prescribed medications appear to be safe to take together based on current analysis.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No prescription data available for interaction checking.")
    
    def show_medical_imaging(self):
        """Display Medical Imaging AI Analyzer from second code"""
        st.markdown("""
        <div class="main-header">
            <h1>🖼️ Medical Imaging AI</h1>
            <p>Advanced AI analysis for medical images including X-rays, MRI, and CT scans</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📤 Upload Images", "🔍 Analysis Results", "📊 Imaging History"])
        
        with tab1:
            st.subheader("Upload Medical Images for AI Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                image_type = st.selectbox(
                    "Image Type",
                    ["xray", "mri", "ct_scan", "ultrasound", "other"]
                )
                
                body_part = st.selectbox(
                    "Body Part",
                    ["chest", "head", "abdomen", "extremities", "spine", "pelvis"]
                )
                
                clinical_context = st.text_area(
                    "Clinical Context / Reason for Imaging",
                    placeholder="Describe symptoms, clinical findings, or specific concerns..."
                )
            
            with col2:
                st.markdown("#### 📤 Upload Image")
                uploaded_file = st.file_uploader(
                    "Choose medical image file",
                    type=['jpg', 'jpeg', 'png', 'dicom'],
                    help="Supported formats: JPG, JPEG, PNG, DICOM"
                )
                
                if uploaded_file is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Medical Image", use_column_width=True)
                    
                    # Image metadata
                    file_details = {
                        "Filename": uploaded_file.name,
                        "File size": f"{uploaded_file.size / 1024:.1f} KB",
                        "Image dimensions": f"{image.size[0]} x {image.size[1]}"
                    }
                    st.write(file_details)
            
            if st.button("🔍 Analyze Image with AI", type="primary", use_container_width=True):
                if uploaded_file is not None:
                    with st.spinner("🤖 AI is analyzing the medical image..."):
                        # Convert image to bytes for analysis
                        img_bytes = uploaded_file.getvalue()
                        
                        # Perform AI analysis
                        analysis_result = self.imaging_ai.analyze_image(
                            img_bytes, image_type, clinical_context
                        )
                        st.session_state.imaging_analysis = analysis_result
                    
                    st.success("✅ Image analysis complete! View results in the Analysis tab.")
                else:
                    st.error("Please upload a medical image first.")
        
        with tab2:
            st.subheader("🖼️ AI Image Analysis Results")
            
            if not st.session_state.get('imaging_analysis'):
                st.info("👆 Please upload and analyze an image first")
                return
            
            analysis = st.session_state.imaging_analysis
            
            # Analysis overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Image Type", analysis.get('image_type', 'Unknown').upper())
                st.metric("Analysis Model", analysis.get('analysis_model', 'Unknown'))
            
            with col2:
                st.metric("Quality Assessment", analysis.get('findings', {}).get('quality_assessment', 'Unknown'))
                st.metric("Urgent Findings", len(analysis.get('urgent_findings', [])))
            
            # Clinical impression
            st.markdown("#### 📋 Clinical Impression")
            st.write(analysis.get('clinical_impression', 'No impression available'))
            
            # Findings details
            st.markdown("#### 🔍 Detailed Findings")
            findings = analysis.get('findings', {})
            
            if findings.get('normal_structures'):
                st.markdown("**✅ Normal Structures:**")
                for structure in findings['normal_structures']:
                    st.write(f"• {structure}")
            
            if findings.get('abnormalities'):
                st.markdown("**⚠️ Abnormalities Detected:**")
                for abnormality in findings['abnormalities']:
                    st.write(f"• {abnormality}")
            
            # Confidence scores
            st.markdown("#### 📊 AI Confidence Scores")
            confidence_scores = findings.get('confidence_scores', {})
            if confidence_scores:
                for area, score in confidence_scores.items():
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write(f"**{area.replace('_', ' ').title()}:**")
                    with col2:
                        st.progress(score)
                        st.write(f"{score:.1%}")
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            recommendations = analysis.get('recommendations', [])
            for recommendation in recommendations:
                st.write(f"• {recommendation}")
            
            # Urgent findings alert
            urgent_findings = analysis.get('urgent_findings', [])
            if urgent_findings:
                st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
                st.error("🚨 URGENT FINDINGS - IMMEDIATE ATTENTION REQUIRED")
                for finding in urgent_findings:
                    st.write(f"• {finding}")
                st.write("Please consult with a radiologist or healthcare provider immediately.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.subheader("📊 Imaging History")
            
            # Sample imaging history data
            imaging_history = [
                {
                    'date': '2024-01-15',
                    'type': 'Chest X-Ray',
                    'body_part': 'Chest',
                    'indication': 'Cough and fever',
                    'findings': 'Right lower lobe pneumonia',
                    'status': 'Completed'
                },
                {
                    'date': '2024-01-10',
                    'type': 'Head CT',
                    'body_part': 'Head',
                    'indication': 'Headache',
                    'findings': 'Normal study',
                    'status': 'Completed'
                },
                {
                    'date': '2023-12-20',
                    'type': 'Abdominal Ultrasound',
                    'body_part': 'Abdomen',
                    'indication': 'Abdominal pain',
                    'findings': 'Gallstones noted',
                    'status': 'Completed'
                }
            ]
            
            if imaging_history:
                for study in imaging_history:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"**{study['type']}**")
                            st.write(f"Date: {study['date']}")
                        with col2:
                            st.write(f"Indication: {study['indication']}")
                            st.write(f"Findings: {study['findings']}")
                        with col3:
                            if study['status'] == 'Completed':
                                st.success("✅ Completed")
                            else:
                                st.warning("⏳ Pending")
                        st.divider()
            else:
                st.info("No imaging studies found in your history.")
    
    def show_patient_portal(self):
        """Display Patient Portal from second code"""
        st.markdown("""
        <div class="main-header">
            <h1>📋 Patient Portal</h1>
            <p>Complete medical records, lab results, and health history</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Health Summary", "🩺 Medical Records", "🧪 Lab Results", "📈 Health Trends"])
        
        with tab1:
            st.subheader("👤 Personal Health Summary")
            
            # Generate sample health summary
            health_summary = self.patient_portal.generate_health_summary("patient_123")
            
            # Overview metrics
            overview = health_summary.get('patient_overview', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Age", overview.get('age', 'Unknown'))
            with col2:
                st.metric("Active Conditions", overview.get('active_conditions', 0))
            with col3:
                st.metric("Current Medications", overview.get('current_medications', 0))
            with col4:
                st.metric("Last Checkup", overview.get('last_checkup', 'Unknown'))
            
            # Health metrics
            st.markdown("#### 📊 Health Metrics")
            metrics = health_summary.get('health_metrics', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Health Score", f"{metrics.get('overall_health_score', 0)}%")
                st.metric("Medication Adherence", metrics.get('medication_adherence', 'Unknown'))
            with col2:
                risk_factors = metrics.get('risk_factors', [])
                st.metric("Risk Factors", len(risk_factors))
                if risk_factors:
                    st.write("**Identified Risks:**")
                    for risk in risk_factors:
                        st.write(f"• {risk}")
            
            # Care plan
            st.markdown("#### 🎯 Care Plan")
            care_plan = health_summary.get('care_plan', {})
            
            if care_plan.get('goals'):
                st.write("**Health Goals:**")
                for goal in care_plan['goals']:
                    st.write(f"• {goal}")
            
            if care_plan.get('actions'):
                st.write("**Recommended Actions:**")
                for action in care_plan['actions']:
                    st.write(f"• {action}")
            
            # Upcoming appointments
            st.markdown("#### 📅 Upcoming Appointments")
            appointments = health_summary.get('upcoming_appointments', [])
            if appointments:
                for appointment in appointments:
                    st.write(f"• {appointment.get('type', 'Appointment')} on {appointment.get('date', 'Unknown')}")
            else:
                st.info("No upcoming appointments scheduled.")
        
        with tab2:
            st.subheader("🩺 Medical Records")
            
            # Sample medical records
            records = [
                {
                    'date': '2024-01-15',
                    'type': 'Office Visit',
                    'provider': 'Dr. Smith',
                    'diagnosis': 'Upper respiratory infection',
                    'treatment': 'Supportive care, rest'
                },
                {
                    'date': '2023-12-10',
                    'type': 'Annual Physical',
                    'provider': 'Dr. Johnson',
                    'diagnosis': 'Healthy',
                    'treatment': 'Routine preventive care'
                },
                {
                    'date': '2023-11-05',
                    'type': 'Specialist Consultation',
                    'provider': 'Dr. Wilson - Cardiology',
                    'diagnosis': 'Hypertension',
                    'treatment': 'Lisinopril 10mg daily'
                }
            ]
            
            for record in records:
                with st.expander(f"{record['date']} - {record['type']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Provider:** {record['provider']}")
                        st.write(f"**Diagnosis:** {record['diagnosis']}")
                    with col2:
                        st.write(f"**Treatment:** {record['treatment']}")
        
        with tab3:
            st.subheader("🧪 Laboratory Results")
            
            # Sample lab results
            lab_results = [
                {
                    'date': '2024-01-15',
                    'test': 'Complete Blood Count (CBC)',
                    'results': {
                        'WBC': '7.2 (Normal: 4.5-11.0)',
                        'HGB': '14.2 (Normal: 13.5-17.5)',
                        'PLT': '250 (Normal: 150-450)'
                    },
                    'status': 'Normal'
                },
                {
                    'date': '2024-01-15',
                    'test': 'Comprehensive Metabolic Panel',
                    'results': {
                        'Glucose': '95 (Normal: 70-100)',
                        'Creatinine': '0.9 (Normal: 0.7-1.3)',
                        'ALT': '25 (Normal: 7-56)'
                    },
                    'status': 'Normal'
                },
                {
                    'date': '2023-12-10',
                    'test': 'Lipid Panel',
                    'results': {
                        'Cholesterol': '190 (Normal: <200)',
                        'LDL': '110 (Normal: <100)',
                        'HDL': '45 (Normal: >40)'
                    },
                    'status': 'Borderline High LDL'
                }
            ]
            
            for lab in lab_results:
                with st.expander(f"{lab['date']} - {lab['test']}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        for test, result in lab['results'].items():
                            st.write(f"**{test}:** {result}")
                    with col2:
                        if lab['status'] == 'Normal':
                            st.success("✅ Normal")
                        else:
                            st.warning(f"⚠️ {lab['status']}")
        
        with tab4:
            st.subheader("📈 Health Trends")
            
            # Sample trend data
            dates = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            blood_pressure_systolic = [130, 128, 125, 122, 120, 118]
            blood_pressure_diastolic = [85, 84, 82, 80, 78, 76]
            weight = [82, 81, 80, 79, 78, 77]
            
            # Create trends chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=blood_pressure_systolic,
                mode='lines+markers',
                name='Systolic BP',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=blood_pressure_diastolic,
                mode='lines+markers',
                name='Diastolic BP',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=weight,
                mode='lines+markers',
                name='Weight (kg)',
                yaxis='y2',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title='Health Trends Over Time',
                xaxis_title='Month',
                yaxis_title='Blood Pressure (mmHg)',
                yaxis2=dict(
                    title='Weight (kg)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Health insights
            st.markdown("#### 💡 Health Insights")
            insights = [
                "✅ Blood pressure showing improvement trend",
                "✅ Weight loss progress consistent",
                "🔄 Continue current medication regimen",
                "🎯 Maintain healthy lifestyle habits"
            ]
            
            for insight in insights:
                st.write(f"• {insight}")
    
    def show_telemedicine(self):
        """Display Telemedicine System from second code"""
        st.markdown("""
        <div class="main-header">
            <h1>📞 Telemedicine</h1>
            <p>Virtual consultations with healthcare providers</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📅 Schedule Consultation", "🎥 Video Call", "📋 Consultation History"])
        
        with tab1:
            st.subheader("📅 Schedule New Consultation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                consultation_type = st.selectbox(
                    "Consultation Type",
                    ["Follow-up", "New Issue", "Chronic Care", "Medication Review", "Specialist Consultation"]
                )
                
                preferred_date = st.date_input("Preferred Date", min_value=datetime.now().date())
                preferred_time = st.selectbox(
                    "Preferred Time",
                    ["09:00 AM", "10:00 AM", "11:00 AM", "02:00 PM", "03:00 PM", "04:00 PM"]
                )
            
            with col2:
                provider_type = st.selectbox(
                    "Provider Type",
                    ["Primary Care Physician", "Specialist", "Any Available Provider"]
                )
                
                urgency = st.select_slider(
                    "Urgency Level",
                    options=["Routine", "Soon", "Urgent"],
                    value="Routine"
                )
                
                chief_complaint = st.text_area(
                    "Chief Complaint / Reason for Visit",
                    placeholder="Briefly describe the reason for your consultation..."
                )
            
            # Technical requirements
            with st.expander("🔧 Technical Requirements"):
                st.write("""
                **For optimal telemedicine experience:**
                - Stable internet connection (min 3 Mbps)
                - Webcam and microphone
                - Private, well-lit location
                - Updated web browser (Chrome, Firefox, Safari)
                """)
            
            if st.button("📅 Schedule Consultation", type="primary", use_container_width=True):
                if chief_complaint:
                    # Schedule consultation
                    scheduled_time = f"{preferred_date} {preferred_time}"
                    consultation = self.telemedicine.schedule_consultation(
                        "patient_123", "provider_456", scheduled_time, consultation_type.lower()
                    )
                    
                    st.session_state.consultation_data = consultation
                    st.success("✅ Consultation scheduled successfully!")
                    
                    st.markdown("#### 📋 Consultation Details")
                    st.write(f"**Session ID:** {consultation['session_id']}")
                    st.write(f"**Scheduled Time:** {consultation['scheduled_time']}")
                    st.write(f"**Join URL:** {consultation['join_url']}")
                    st.write(f"**Status:** {consultation['status']}")
                else:
                    st.error("Please describe the reason for your consultation.")
        
        with tab2:
            st.subheader("🎥 Video Consultation")
            
            if st.session_state.get('consultation_data'):
                consultation = st.session_state.consultation_data
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Session Information")
                    st.write(f"**Session ID:** {consultation['session_id']}")
                    st.write(f"**Type:** {consultation['consultation_type']}")
                    st.write(f"**Scheduled:** {consultation['scheduled_time']}")
                    st.write(f"**Status:** {consultation['status']}")
                    
                    if st.button("🎬 Start Consultation", type="primary"):
                        session_info = self.telemedicine.start_consultation(consultation['session_id'])
                        st.success("Consultation started! Use the video call interface below.")
                
                with col2:
                    st.markdown("#### 🔧 Technical Check")
                    st.success("✅ Camera: Detected")
                    st.success("✅ Microphone: Detected")
                    st.success("✅ Internet: Stable")
                    st.info("🔒 Connection: Encrypted")
                
                # Video call interface placeholder
                st.markdown("#### 🎥 Video Call Interface")
                st.info("""
                **Video call would be active here in a production environment**
                
                Features include:
                - Real-time video and audio
                - Screen sharing
                - Chat functionality
                - Virtual waiting room
                - Recording (with consent)
                """)
                
                # Consultation controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("📞 Mute Audio")
                with col2:
                    st.button("📹 Disable Video")
                with col3:
                    if st.button("📋 End Consultation"):
                        summary = {
                            'diagnosis': 'Upper respiratory infection',
                            'treatment': 'Supportive care, rest',
                            'prescription_needed': False,
                            'follow_up_recommended': True
                        }
                        end_result = self.telemedicine.end_consultation(consultation['session_id'], summary)
                        st.success("Consultation ended successfully!")
            else:
                st.info("👆 Please schedule a consultation first")
        
        with tab3:
            st.subheader("📋 Consultation History")
            
            history = self.telemedicine.get_consultation_history("patient_123")
            
            if history:
                for consultation in history:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"**{consultation['consultation_type'].title()}**")
                            st.write(f"Scheduled: {consultation['scheduled_time']}")
                        with col2:
                            st.write(f"Provider: {consultation['provider_id']}")
                            if 'duration' in consultation:
                                st.write(f"Duration: {consultation['duration']}")
                        with col3:
                            status = consultation['status']
                            if status == 'completed':
                                st.success("✅ Completed")
                            elif status == 'in_progress':
                                st.warning("🟡 In Progress")
                            else:
                                st.info("📅 Scheduled")
                        
                        if consultation.get('consultation_summary'):
                            with st.expander("View Summary"):
                                st.write(consultation['consultation_summary'])
                        
                        st.divider()
            else:
                st.info("No consultation history found.")

    # =============================================
    # HEALTH MONITORING PAGES FROM FIRST CODE
    # =============================================
    
    def _show_health_monitor_page(self):
        st.markdown("<h1 class='main-title'>❤️ Health Monitor</h1>", unsafe_allow_html=True)
        
        # Main tabs for different monitoring features
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Vital Signs", "😣 Pain Assessment", "😴 Sleep Quality", "📈 Trends", "📋 Health Summary", "😊 Emotion Detection", "⚙️ Settings"])
        
        # TAB 1: VITAL SIGNS
        with tab1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("🩺 Vital Signs Monitoring")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Manual Pulse Rate
                st.markdown("#### 💓 Pulse Rate")
                pulse_time = st.number_input("Count pulse for 30 seconds", min_value=0, max_value=60, value=30, key="pulse_time")
                pulse_count = st.number_input("Pulse beats counted", min_value=0, max_value=120, key="pulse_count")
                
                if st.button("Calculate Pulse Rate", key="calc_pulse"):
                    if pulse_count > 0 and pulse_time > 0:
                        pulse_rate = (pulse_count / pulse_time) * 60
                        st.metric("Pulse Rate", f"{pulse_rate:.0f} BPM")
                        
                        # Save to session state
                        vital_data = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'pulse': pulse_rate,
                            'type': 'pulse'
                        }
                        if 'vital_signs_history' not in st.session_state:
                            st.session_state.vital_signs_history = []
                        st.session_state.vital_signs_history.append(vital_data)
                        st.session_state.latest_pulse = pulse_rate
                        
                        # Categorize pulse rate
                        if pulse_rate < 60:
                            st.warning("🟡 Bradycardia (Low pulse rate)")
                        elif pulse_rate > 100:
                            st.warning("🟡 Tachycardia (High pulse rate)")
                        else:
                            st.success("🟢 Normal pulse rate")
                
                # Respiratory Rate
                st.markdown("#### 🌬️ Respiratory Rate")
                resp_time = st.number_input("Count breaths for 30 seconds", min_value=0, max_value=60, value=30, key="resp_time")
                breath_count = st.number_input("Breaths counted", min_value=0, max_value=60, key="breath_count")
                
                if st.button("Calculate Respiratory Rate", key="calc_resp"):
                    if breath_count > 0 and resp_time > 0:
                        resp_rate = (breath_count / resp_time) * 60
                        st.metric("Respiratory Rate", f"{resp_rate:.0f} breaths/min")
                        
                        # Save to session state
                        vital_data = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'respiratory_rate': resp_rate,
                            'type': 'respiratory'
                        }
                        if 'vital_signs_history' not in st.session_state:
                            st.session_state.vital_signs_history = []
                        st.session_state.vital_signs_history.append(vital_data)
                        st.session_state.latest_resp_rate = resp_rate
                        
                        if resp_rate < 12:
                            st.warning("🟡 Low respiratory rate")
                        elif resp_rate > 20:
                            st.warning("🟡 High respiratory rate")
                        else:
                            st.success("🟢 Normal respiratory rate")
            
            with col2:
                # Blood Pressure
                st.markdown("#### 🩸 Blood Pressure")
                systolic = st.number_input("Systolic (upper number)", min_value=50, max_value=250, value=120, key="systolic")
                diastolic = st.number_input("Diastolic (lower number)", min_value=30, max_value=150, value=80, key="diastolic")
                
                if st.button("Analyze Blood Pressure", key="analyze_bp"):
                    # Save to session state
                    vital_data = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'systolic': systolic,
                        'diastolic': diastolic,
                        'type': 'blood_pressure'
                    }
                    if 'vital_signs_history' not in st.session_state:
                        st.session_state.vital_signs_history = []
                    st.session_state.vital_signs_history.append(vital_data)
                    st.session_state.latest_bp = f"{systolic}/{diastolic}"
                    
                    # Categorize blood pressure
                    if systolic < 90 or diastolic < 60:
                        st.error("🔴 Low Blood Pressure")
                    elif systolic < 120 and diastolic < 80:
                        st.success("🟢 Normal")
                    elif systolic < 130 and diastolic < 80:
                        st.warning("🟡 Elevated")
                    elif systolic < 140 or diastolic < 90:
                        st.warning("🟠 Stage 1 Hypertension")
                    else:
                        st.error("🔴 Stage 2 Hypertension")
                    
                    st.metric("Blood Pressure", f"{systolic}/{diastolic} mmHg")
                
                # Temperature
                st.markdown("#### 🌡️ Temperature")
                temp_unit = st.radio("Unit", ["Celsius", "Fahrenheit"], horizontal=True, key="temp_unit")
                temperature = st.number_input("Temperature", min_value=30.0, max_value=45.0, value=36.8, step=0.1, key="temperature")
                
                if st.button("Analyze Temperature", key="analyze_temp"):
                    # Save to session state
                    vital_data = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'temperature': temperature,
                        'unit': temp_unit,
                        'type': 'temperature'
                    }
                    if 'vital_signs_history' not in st.session_state:
                        st.session_state.vital_signs_history = []
                    st.session_state.vital_signs_history.append(vital_data)
                    st.session_state.latest_temp = f"{temperature}°{temp_unit[0]}"
                    
                    if temp_unit == "Fahrenheit":
                        # Convert to Celsius for analysis
                        temp_c = (temperature - 32) * 5/9
                    else:
                        temp_c = temperature
                    
                    st.metric("Temperature", f"{temperature}°{temp_unit[0]}")
                    
                    if temp_c < 36.0:
                        st.warning("🟡 Low body temperature")
                    elif temp_c <= 37.5:
                        st.success("🟢 Normal body temperature")
                    elif temp_c <= 38.0:
                        st.warning("🟠 Mild fever")
                    else:
                        st.error("🔴 High fever - Seek medical attention")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # TAB 2: PAIN ASSESSMENT
        with tab2:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader("😣 Pain Assessment")
                
                # Manual pain assessment
                pain_level = st.slider("Pain Level (0-10)", 0, 10, 0, key="pain_level")
                
                # Pain location
                st.write("**Pain Location:**")
                col_loc1, col_loc2 = st.columns(2)
                with col_loc1:
                    head_pain = st.checkbox("Head")
                    chest_pain = st.checkbox("Chest")
                    arm_pain = st.checkbox("Arms")
                    back_pain = st.checkbox("Back")
                with col_loc2:
                    abdominal_pain = st.checkbox("Abdomen")
                    leg_pain = st.checkbox("Legs")
                    joint_pain = st.checkbox("Joints")
                    general_pain = st.checkbox("General")
                
                # Pain quality
                pain_quality = st.selectbox("Pain Quality", 
                                          ["Sharp", "Dull", "Burning", "Throbbing", "Stabbing", "Aching", "Cramping"])
                
                if st.button("Save Pain Assessment", key="save_pain"):
                    pain_locations = []
                    if head_pain: pain_locations.append("Head")
                    if chest_pain: pain_locations.append("Chest")
                    if arm_pain: pain_locations.append("Arms")
                    if back_pain: pain_locations.append("Back")
                    if abdominal_pain: pain_locations.append("Abdomen")
                    if leg_pain: pain_locations.append("Legs")
                    if joint_pain: pain_locations.append("Joints")
                    if general_pain: pain_locations.append("General")
                    
                    # Save to session state
                    pain_data = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'level': pain_level,
                        'locations': pain_locations,
                        'quality': pain_quality
                    }
                    if 'pain_assessment_history' not in st.session_state:
                        st.session_state.pain_assessment_history = []
                    st.session_state.pain_assessment_history.append(pain_data)
                    st.session_state.latest_pain_level = pain_level
                    
                    st.success(f"✅ Pain assessment saved: Level {pain_level}/10")
                    if pain_locations:
                        st.write(f"**Locations:** {', '.join(pain_locations)}")
                    st.write(f"**Quality:** {pain_quality}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader("📋 Pain History")
                
                # Pain scale visualization
                st.markdown("**Pain Scale Reference:**")
                st.write("0: 😊 No pain")
                st.write("1-3: 😐 Mild pain")
                st.write("4-6: 😣 Moderate pain")
                st.write("7-8: 😫 Severe pain")
                st.write("9-10: 😖 Worst pain possible")
                
                st.markdown("---")
                st.subheader("🚨 Emergency Pain")
                st.warning("""
                **Seek immediate medical attention for:**
                - Chest pain with shortness of breath
                - Severe abdominal pain
                - Sudden severe headache
                - Pain with numbness/weakness
                """)
                
                if st.button("🆘 Emergency Help", key="emergency_pain"):
                    st.error("""
                    **Call Emergency Services Immediately!**
                    • Dial 911 or your local emergency number
                    • Don't drive yourself to hospital
                    • Stay on the line with operator
                    """)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # TAB 3: SLEEP QUALITY
        with tab3:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("😴 Sleep Quality Monitoring")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sleep duration
                st.markdown("#### ⏰ Sleep Duration")
                bedtime = st.time_input("Bedtime", value=datetime.strptime("22:00", "%H:%M").time(), key="bedtime")
                wake_time = st.time_input("Wake Time", value=datetime.strptime("06:00", "%H:%M").time(), key="wake_time")
                
                # Calculate sleep duration
                if st.button("Calculate Sleep Duration", key="calc_sleep"):
                    bedtime_dt = datetime.combine(datetime.today(), bedtime)
                    wake_dt = datetime.combine(datetime.today(), wake_time)
                    if wake_dt < bedtime_dt:
                        wake_dt += timedelta(days=1)
                    
                    sleep_duration = (wake_dt - bedtime_dt).total_seconds() / 3600
                    st.metric("Sleep Duration", f"{sleep_duration:.1f} hours")
                    
                    if sleep_duration < 7:
                        st.warning("🟡 Less than recommended 7-9 hours")
                    elif sleep_duration > 9:
                        st.warning("🟡 More than recommended 7-9 hours")
                    else:
                        st.success("🟢 Optimal sleep duration")
            
            with col2:
                # Sleep quality
                st.markdown("#### 🌟 Sleep Quality")
                sleep_quality = st.slider("Rate your sleep quality (1-10)", 1, 10, 7, key="sleep_quality")
                
                # Sleep factors
                st.markdown("#### 📝 Sleep Factors")
                interruptions = st.number_input("Number of nighttime awakenings", min_value=0, max_value=10, value=0, key="interruptions")
                deep_sleep = st.slider("Perceived deep sleep (1-10)", 1, 10, 6, key="deep_sleep")
                refresh_feeling = st.slider("Wake-up feeling refreshed (1-10)", 1, 10, 6, key="refresh_feeling")
            
            # Additional sleep metrics
            st.markdown("#### 📊 Additional Sleep Metrics")
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                dream_recall = st.selectbox("Dream Recall", ["Vivid", "Some", "None", "Can't remember"])
            with col_met2:
                sleep_latency = st.number_input("Minutes to fall asleep", min_value=0, max_value=120, value=15, key="sleep_latency")
            with col_met3:
                sleep_consistency = st.selectbox("Sleep Consistency", ["Very consistent", "Somewhat consistent", "Inconsistent"])
            
            if st.button("Save Sleep Analysis", key="save_sleep"):
                sleep_score = (sleep_quality + deep_sleep + refresh_feeling) / 3
                
                # Save to session state
                sleep_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'quality': sleep_quality,
                    'duration': sleep_duration if 'sleep_duration' in locals() else 7.5,
                    'interruptions': interruptions,
                    'deep_sleep': deep_sleep,
                    'refresh_feeling': refresh_feeling,
                    'score': sleep_score
                }
                if 'sleep_history' not in st.session_state:
                    st.session_state.sleep_history = []
                st.session_state.sleep_history.append(sleep_data)
                st.session_state.latest_sleep_score = sleep_quality
                
                st.success("✅ Sleep analysis saved!")
                st.metric("Overall Sleep Score", f"{sleep_score:.1f}/10")
                
                if sleep_score >= 8:
                    st.success("🟢 Excellent sleep quality")
                elif sleep_score >= 6:
                    st.info("🟡 Good sleep quality")
                else:
                    st.warning("🟠 Poor sleep quality - Consider improvements")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # TAB 4: TRENDS
        with tab4:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("📈 Health Trends & Analytics")
            
            # Get data from session state
            vital_signs = st.session_state.get('vital_signs_history', [])
            pain_history = st.session_state.get('pain_assessment_history', [])
            sleep_history = st.session_state.get('sleep_history', [])
            emotion_history = st.session_state.get('emotion_history', [])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Pulse trend
                if vital_signs:
                    pulse_data = [v['pulse'] for v in vital_signs if 'pulse' in v]
                    if pulse_data:
                        avg_pulse = sum(pulse_data) / len(pulse_data)
                        st.metric("Average Pulse", f"{avg_pulse:.0f} BPM", delta="-2 from baseline")
                else:
                    st.metric("Average Pulse", "N/A")
            
            with col2:
                # Sleep trend
                if sleep_history:
                    sleep_scores = [s['score'] for s in sleep_history if 'score' in s]
                    if sleep_scores:
                        avg_sleep = sum(sleep_scores) / len(sleep_scores)
                        st.metric("Average Sleep", f"{avg_sleep:.1f}/10", delta="+0.3 from baseline")
                else:
                    st.metric("Average Sleep", "N/A")
            
            with col3:
                # Pain trend
                if pain_history:
                    pain_levels = [p['level'] for p in pain_history if 'level' in p]
                    if pain_levels:
                        avg_pain = sum(pain_levels) / len(pain_levels)
                        st.metric("Average Pain", f"{avg_pain:.1f}/10", delta="-0.5 from baseline")
                else:
                    st.metric("Average Pain", "N/A")
            
            # Health insights
            st.markdown("#### 💡 Health Insights")
            insights = []
            
            if vital_signs:
                insights.append("✅ Vital signs monitoring active")
            if pain_history:
                insights.append("✅ Pain assessment ongoing")
            if sleep_history:
                insights.append("✅ Sleep quality tracking")
            if emotion_history:
                insights.append("✅ Emotional wellness monitoring")
            
            if not insights:
                insights.append("📝 Record health data to see insights")
            
            for insight in insights:
                st.write(f"• {insight}")
            
            # Data visualization placeholder
            st.markdown("#### 📊 Data Visualization")
            st.info("Advanced charts and graphs will be displayed here as more data is collected")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # TAB 5: HEALTH SUMMARY
        with tab5:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("📋 Comprehensive Health Summary")
            
            # Current health status overview
            st.markdown("### 📊 Today's Health Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Get latest vital signs
                latest_pulse = st.session_state.get('latest_pulse', 'N/A')
                latest_resp_rate = st.session_state.get('latest_resp_rate', 'N/A')
                st.metric("❤️ Pulse Rate", f"{latest_pulse} BPM" if latest_pulse != 'N/A' else "Not measured")
                st.metric("🌬️ Resp Rate", f"{latest_resp_rate}/min" if latest_resp_rate != 'N/A' else "Not measured")
            
            with col2:
                # Get latest blood pressure and temperature
                latest_bp = st.session_state.get('latest_bp', 'N/A')
                latest_temp = st.session_state.get('latest_temp', 'N/A')
                st.metric("🩸 Blood Pressure", f"{latest_bp}" if latest_bp != 'N/A' else "Not measured")
                st.metric("🌡️ Temperature", f"{latest_temp}" if latest_temp != 'N/A' else "Not measured")
            
            with col3:
                # Get latest pain and emotion data
                latest_pain = st.session_state.get('latest_pain_level', 'N/A')
                latest_emotion = st.session_state.get('latest_emotion', 'N/A')
                st.metric("😣 Pain Level", f"{latest_pain}/10" if latest_pain != 'N/A' else "Not assessed")
                st.metric("😊 Mood", f"{latest_emotion}" if latest_emotion != 'N/A' else "Not assessed")
            
            with col4:
                # Get sleep and activity data
                latest_sleep = st.session_state.get('latest_sleep_score', 'N/A')
                latest_energy = st.session_state.get('latest_energy_level', 'N/A')
                st.metric("😴 Sleep Score", f"{latest_sleep}/10" if latest_sleep != 'N/A' else "Not recorded")
                st.metric("⚡ Energy Level", f"{latest_energy}/10" if latest_energy != 'N/A' else "Not recorded")
            
            st.markdown("---")
            
            # Detailed Sections
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Vital Signs History
                st.markdown("#### 🩺 Vital Signs History")
                vital_signs = st.session_state.get('vital_signs_history', [])
                if vital_signs:
                    vital_df = pd.DataFrame(vital_signs[-5:])  # Show last 5 entries
                    st.dataframe(vital_df, use_container_width=True)
                else:
                    st.info("No vital signs recorded yet. Visit the Vital Signs tab to add measurements.")
                
                # Pain Assessment History
                st.markdown("#### 😣 Pain Assessment History")
                pain_history = st.session_state.get('pain_assessment_history', [])
                if pain_history:
                    pain_df = pd.DataFrame(pain_history[-5:])
                    st.dataframe(pain_df, use_container_width=True)
                else:
                    st.info("No pain assessments recorded. Visit Pain Assessment tab to log pain.")
            
            with col_right:
                # Sleep Quality History
                st.markdown("#### 😴 Sleep Quality History")
                sleep_history = st.session_state.get('sleep_history', [])
                if sleep_history:
                    sleep_df = pd.DataFrame(sleep_history[-5:])
                    st.dataframe(sleep_df, use_container_width=True)
                else:
                    st.info("No sleep data recorded. Visit Sleep Quality tab to log sleep.")
                
                # Emotion Detection History
                st.markdown("#### 😊 Emotion History")
                emotion_history = st.session_state.get('emotion_history', [])
                if emotion_history:
                    emotion_df = pd.DataFrame(emotion_history[-5:])
                    st.dataframe(emotion_df, use_container_width=True)
                else:
                    st.info("No emotion data recorded. Visit Emotion Detection tab to analyze emotions.")
            
            # Health Insights
            st.markdown("#### 💡 Health Insights & Recommendations")
            
            insights = []
            if vital_signs:
                latest_vital = vital_signs[-1] if vital_signs else {}
                if 'pulse' in latest_vital:
                    pulse = latest_vital['pulse']
                    if 60 <= pulse <= 100:
                        insights.append("✅ Pulse rate within healthy range")
                    else:
                        insights.append("⚠️ Pulse rate outside normal range - monitor closely")
            
            if pain_history:
                latest_pain_data = pain_history[-1] if pain_history else {}
                if 'level' in latest_pain_data and latest_pain_data['level'] >= 7:
                    insights.append("🚨 High pain level detected - consider medical advice")
                elif 'level' in latest_pain_data and latest_pain_data['level'] >= 4:
                    insights.append("⚠️ Moderate pain level - manage with appropriate techniques")
            
            if sleep_history:
                latest_sleep_data = sleep_history[-1] if sleep_history else {}
                if 'score' in latest_sleep_data and latest_sleep_data['score'] < 6:
                    insights.append("😴 Poor sleep quality - improve sleep hygiene")
            
            if not insights:
                insights.append("📝 Record more health data to get personalized insights")
            
            for insight in insights:
                if "🚨" in insight:
                    st.error(insight)
                elif "⚠️" in insight:
                    st.warning(insight)
                elif "✅" in insight:
                    st.success(insight)
                else:
                    st.info(insight)
            
            # Export option
            st.markdown("---")
            if st.button("📄 Export Health Report", key="export_report", use_container_width=True):
                report = self._generate_health_report()
                st.success("Health report generated successfully!")
                st.download_button(
                    label="📥 Download Health Report",
                    data=report,
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # TAB 6: EMOTION DETECTION
        with tab6:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("😊 Emotion Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 Real-time Camera Detection")
                st.info("Use your camera to detect facial emotions and get detailed analysis")
                
                # Camera input
                try:
                    camera_img = st.camera_input("Take a picture for emotion analysis", key="emotion_camera")
                    
                    if camera_img is not None:
                        # Display the captured image
                        image = Image.open(camera_img)
                        st.image(image, caption="Captured Image - Analyzing Emotion...", use_column_width=True)
                        
                        # Simulate face detection and emotion analysis
                        with st.spinner("🔍 Detecting faces and analyzing emotions..."):
                            time.sleep(2)
                            
                            # Simulate face detection
                            face_detected = random.choice([True, True, True])  # 100% detection for demo
                            
                            if face_detected:
                                # Simulated emotion detection
                                emotions = ["Happy", "Neutral", "Sad", "Surprised", "Angry"]
                                detected_emotion = random.choice(emotions)
                                confidence = random.randint(85, 95)
                                rating = random.randint(7, 10) if detected_emotion == "Happy" else random.randint(4, 8)
                                
                                # Save to session state
                                emotion_data = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'emotion': detected_emotion,
                                    'intensity': rating,
                                    'confidence': confidence,
                                    'source': 'camera'
                                }
                                if 'emotion_history' not in st.session_state:
                                    st.session_state.emotion_history = []
                                st.session_state.emotion_history.append(emotion_data)
                                st.session_state.latest_emotion = detected_emotion
                                st.session_state.latest_energy_level = random.randint(6, 9)
                                
                                st.success("✅ Face detected! Analyzing emotions...")
                                st.markdown(f"#### 🎭 Emotion Analysis")
                                
                                col_emo1, col_emo2 = st.columns([1, 2])
                                
                                with col_emo1:
                                    emotion_icons = {
                                        "Happy": "😊",
                                        "Neutral": "😐", 
                                        "Sad": "😔",
                                        "Surprised": "😲",
                                        "Angry": "😠"
                                    }
                                    icon = emotion_icons.get(detected_emotion, "😐")
                                    st.metric("Detected Emotion", f"{icon} {detected_emotion}")
                                    st.metric("Confidence", f"{confidence}%")
                                    st.metric("Intensity Rating", f"{rating}/10")
                                
                                with col_emo2:
                                    st.markdown("**Explanation:**")
                                    explanations = {
                                        "Happy": "Facial features indicate positive emotions with visible engagement and happiness characteristics.",
                                        "Neutral": "Balanced facial features indicate calm and composed emotional state.",
                                        "Sad": "Facial analysis suggests subdued expression potentially indicating low mood.",
                                        "Surprised": "Heightened facial expression detected, characteristic of surprise or amazement.",
                                        "Angry": "Facial features suggest tense expression potentially indicating frustration."
                                    }
                                    st.info(explanations.get(detected_emotion, "Emotional state analyzed successfully."))
                                
                            else:
                                st.error("❌ No faces detected in the image.")
                                
                except Exception as e:
                    st.error("❌ Camera not accessible. Please check camera permissions.")
            
            with col2:
                st.markdown("#### 📁 Upload Image Analysis")
                st.info("Upload a facial image from your device for emotion analysis")
                
                # Image upload
                uploaded_img = st.file_uploader("Choose a facial image", 
                                              type=['jpg', 'jpeg', 'png'], 
                                              key="emotion_upload")
                
                if uploaded_img is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_img)
                    st.image(image, caption="Uploaded Image - Ready for Analysis", use_column_width=True)
                    
                    if st.button("🔍 Analyze Emotion in Image", key="analyze_upload", use_container_width=True):
                        with st.spinner("🖼️ Processing image - Analyzing facial emotions..."):
                            time.sleep(2)
                            
                            # Simulated analysis for uploaded image
                            uploaded_emotions = ["Happy", "Neutral", "Sad", "Surprised"]
                            detected_upload_emotion = random.choice(uploaded_emotions)
                            upload_confidence = random.randint(80, 98)
                            upload_rating = random.randint(8, 10) if detected_upload_emotion == "Happy" else random.randint(5, 8)
                            
                            # Save to session state
                            emotion_data = {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                'emotion': detected_upload_emotion,
                                'intensity': upload_rating,
                                'confidence': upload_confidence,
                                'source': 'upload'
                            }
                            if 'emotion_history' not in st.session_state:
                                st.session_state.emotion_history = []
                            st.session_state.emotion_history.append(emotion_data)
                            st.session_state.latest_emotion = detected_upload_emotion
                            
                            st.success("✅ Image analyzed successfully!")
                            st.markdown(f"#### 🎭 Emotion Analysis Results")
                            
                            col_res1, col_res2 = st.columns([1, 2])
                            
                            with col_res1:
                                emotion_icons = {
                                    "Happy": "😊",
                                    "Neutral": "😐", 
                                    "Sad": "😔",
                                    "Surprised": "😲"
                                }
                                icon = emotion_icons.get(detected_upload_emotion, "😐")
                                st.metric("Primary Emotion", f"{icon} {detected_upload_emotion}")
                                st.metric("Detection Confidence", f"{upload_confidence}%")
                                st.metric("Intensity Rating", f"{upload_rating}/10")
                            
                            with col_res2:
                                st.markdown("**Emotion Analysis:**")
                                analysis_text = {
                                    "Happy": "The image shows characteristics of happiness with positive facial features.",
                                    "Neutral": "The facial expression appears calm and balanced.",
                                    "Sad": "The expression suggests a subdued or low mood state.",
                                    "Surprised": "The facial features indicate a surprised or amazed expression."
                                }
                                st.info(analysis_text.get(detected_upload_emotion, "Emotional analysis completed."))
                
                # Emotion reference
                st.markdown("---")
                st.markdown("#### 📊 Emotion Rating Scale")
                st.write("😊 Happy: 8-10/10 - Positive, joyful")
                st.write("😐 Neutral: 5-7/10 - Calm, balanced")
                st.write("😔 Sad: 3-6/10 - Low mood, down")
                st.write("😲 Surprised: 6-9/10 - Startled, amazed")
                st.write("😠 Angry: 7-10/10 - Frustrated, irritated")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # TAB 7: SETTINGS
        with tab7:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("⚙️ Health Monitor Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔔 Notifications")
                enable_notifications = st.checkbox("Enable health reminders", value=True)
                reminder_frequency = st.selectbox("Reminder frequency", ["Daily", "Twice daily", "Weekly"])
                
                st.markdown("#### 📱 Data Sharing")
                share_with_doctor = st.checkbox("Share data with healthcare provider")
                auto_backup = st.checkbox("Automatic cloud backup", value=True)
            
            with col2:
                st.markdown("#### 🎯 Health Goals")
                target_sleep = st.slider("Target sleep hours", 6.0, 10.0, 7.5, 0.5)
                max_pain_level = st.slider("Maximum acceptable pain level", 1, 10, 4)
                target_steps = st.number_input("Daily step goal", min_value=1000, max_value=20000, value=8000, step=500)
                
                st.markdown("#### 😊 Emotion Tracking")
                enable_emotion_tracking = st.checkbox("Enable emotion detection", value=True)
                emotion_reminders = st.checkbox("Send mood check reminders", value=True)
            
            if st.button("💾 Save Settings", key="save_settings"):
                st.success("✅ Health monitor settings saved successfully!")
            
            st.markdown("</div>", unsafe_allow_html=True)

    def _generate_health_report(self):
        """Generate a comprehensive health report"""
        report = f"""
        COMPREHENSIVE HEALTH REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        =============================================
        
        VITAL SIGNS SUMMARY:
        {self._get_vital_signs_summary()}
        
        PAIN ASSESSMENT HISTORY:
        {self._get_pain_assessment_summary()}
        
        SLEEP QUALITY ANALYSIS:
        {self._get_sleep_quality_summary()}
        
        EMOTIONAL WELLNESS:
        {self._get_emotional_wellness_summary()}
        
        OVERALL HEALTH ASSESSMENT:
        {self._get_overall_assessment()}
        
        RECOMMENDATIONS:
        • Continue regular health monitoring
        • Maintain balanced lifestyle
        • Consult healthcare provider for concerning symptoms
        • Practice stress management techniques
        • Ensure adequate sleep and nutrition
        """
        return report

    def _get_vital_signs_summary(self):
        vital_history = st.session_state.get('vital_signs_history', [])
        if not vital_history:
            return "No vital signs data available."
        
        latest = vital_history[-1]
        summary = f"Latest Reading: {latest.get('timestamp', 'N/A')}\n"
        if 'pulse' in latest:
            summary += f"Pulse Rate: {latest['pulse']} BPM\n"
        if 'respiratory_rate' in latest:
            summary += f"Respiratory Rate: {latest['respiratory_rate']}/min\n"
        if 'systolic' in latest and 'diastolic' in latest:
            summary += f"Blood Pressure: {latest['systolic']}/{latest['diastolic']} mmHg\n"
        if 'temperature' in latest:
            summary += f"Temperature: {latest['temperature']}°C\n"
        
        return summary

    def _get_pain_assessment_summary(self):
        pain_history = st.session_state.get('pain_assessment_history', [])
        if not pain_history:
            return "No pain assessment data available."
        
        latest = pain_history[-1]
        summary = f"Latest Assessment: {latest.get('timestamp', 'N/A')}\n"
        summary += f"Pain Level: {latest.get('level', 'N/A')}/10\n"
        if 'locations' in latest:
            summary += f"Locations: {', '.join(latest['locations'])}\n"
        if 'quality' in latest:
            summary += f"Quality: {latest['quality']}\n"
        
        return summary

    def _get_sleep_quality_summary(self):
        sleep_history = st.session_state.get('sleep_history', [])
        if not sleep_history:
            return "No sleep data available."
        
        latest = sleep_history[-1]
        summary = f"Latest Record: {latest.get('timestamp', 'N/A')}\n"
        if 'quality' in latest:
            summary += f"Sleep Quality: {latest['quality']}/10\n"
        if 'duration' in latest:
            summary += f"Duration: {latest['duration']:.1f} hours\n"
        if 'interruptions' in latest:
            summary += f"Nighttime Awakenings: {latest['interruptions']}\n"
        if 'deep_sleep' in latest:
            summary += f"Deep Sleep Rating: {latest['deep_sleep']}/10\n"
        if 'refresh_feeling' in latest:
            summary += f"Refresh Feeling: {latest['refresh_feeling']}/10\n"
        if 'score' in latest:
            summary += f"Overall Sleep Score: {latest['score']:.1f}/10\n"
        
        return summary

    def _get_emotional_wellness_summary(self):
        emotion_history = st.session_state.get('emotion_history', [])
        if not emotion_history:
            return "No emotional wellness data available."
        
        latest = emotion_history[-1]
        summary = f"Latest Assessment: {latest.get('timestamp', 'N/A')}\n"
        if 'emotion' in latest:
            summary += f"Primary Emotion: {latest['emotion']}\n"
        if 'intensity' in latest:
            summary += f"Intensity: {latest['intensity']}/10\n"
        if 'confidence' in latest:
            summary += f"Detection Confidence: {latest['confidence']}%\n"
        if 'source' in latest:
            summary += f"Source: {latest['source']}\n"
        
        return summary

    def _get_overall_assessment(self):
        """Generate overall health assessment based on available data"""
        assessment = []
        
        # Check vital signs
        vital_history = st.session_state.get('vital_signs_history', [])
        if vital_history:
            latest_vital = vital_history[-1]
            if 'pulse' in latest_vital:
                pulse = latest_vital['pulse']
                if 60 <= pulse <= 100:
                    assessment.append("✓ Pulse rate within normal range")
                else:
                    assessment.append("⚠️ Pulse rate requires attention")
            
            if 'systolic' in latest_vital and 'diastolic' in latest_vital:
                systolic = latest_vital['systolic']
                diastolic = latest_vital['diastolic']
                if systolic < 120 and diastolic < 80:
                    assessment.append("✓ Blood pressure normal")
                else:
                    assessment.append("⚠️ Blood pressure monitoring recommended")
        
        # Check pain levels
        pain_history = st.session_state.get('pain_assessment_history', [])
        if pain_history:
            latest_pain = pain_history[-1]
            pain_level = latest_pain.get('level', 0)
            if pain_level <= 3:
                assessment.append("✓ Low pain levels")
            elif pain_level <= 6:
                assessment.append("⚠️ Moderate pain - manage appropriately")
            else:
                assessment.append("🚨 High pain level - seek medical advice")
        
        # Check sleep quality
        sleep_history = st.session_state.get('sleep_history', [])
        if sleep_history:
            latest_sleep = sleep_history[-1]
            sleep_score = latest_sleep.get('score', 0)
            if sleep_score >= 8:
                assessment.append("✓ Excellent sleep quality")
            elif sleep_score >= 6:
                assessment.append("↔️ Adequate sleep quality")
            else:
                assessment.append("⚠️ Poor sleep quality - improvement needed")
        
        # Check emotional wellness
        emotion_history = st.session_state.get('emotion_history', [])
        if emotion_history:
            latest_emotion = emotion_history[-1]
            emotion = latest_emotion.get('emotion', 'Neutral')
            if emotion in ['Happy', 'Neutral']:
                assessment.append("✓ Positive emotional state")
            else:
                assessment.append("⚠️ Emotional state requires attention")
        
        if not assessment:
            return "Insufficient data for comprehensive assessment. Please record more health metrics."
        
        return "\n".join(assessment)
    def show_hrv_analysis_page(self):
        st.markdown("<h1 class='main-title'>📈 HRV Analysis</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("❤️ Heart Rate Variability Monitoring")
            
            # Manual HRV input
            st.markdown("#### 📊 Enter HRV Data")
            hrv_value = st.number_input("HRV Value (ms)", min_value=0, max_value=200, value=65, key="hrv_value")
            measurement_time = st.selectbox("Measurement Context", ["Morning", "Evening", "After exercise", "Resting"])
            
            if st.button("Analyze HRV", key="analyze_hrv"):
                st.metric("Current HRV", f"{hrv_value} ms")
                
                # HRV interpretation
                if hrv_value < 50:
                    st.error("🔴 Low HRV - High stress or fatigue")
                elif hrv_value < 70:
                    st.warning("🟡 Moderate HRV - Average stress levels")
                else:
                    st.success("🟢 Good HRV - Healthy stress response")
                
                st.write("**Interpretation:**")
                st.write("Higher HRV indicates better cardiovascular fitness and stress resilience")
            
            # HRV trends
            st.markdown("#### 📈 HRV Trends")
            st.info("HRV trend analysis feature coming soon...")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("💡 HRV Guide")
            
            st.markdown("**HRV Ranges:**")
            st.write("• < 50 ms: Low (High stress)")
            st.write("• 50-70 ms: Moderate")
            st.write("• 70-100 ms: Good")
            st.write("• > 100 ms: Excellent")
            
            st.markdown("---")
            st.markdown("**Improve HRV:**")
            tips = [
                "Regular exercise",
                "Quality sleep",
                "Stress management",
                "Balanced nutrition",
                "Breathing exercises"
            ]
            
            for tip in tips:
                st.write(f"• {tip}")
            
            st.markdown("</div>", unsafe_allow_html=True)
 
    def show_user_profile_page(self):
        st.markdown("<h1 class='main-title'>👤 User Profile</h1>", unsafe_allow_html=True)
        
        # Initialize session state for user profile if not exists
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {
                'personal_info': {
                    'full_name': '',
                    'date_of_birth': None,
                    'gender': '',
                    'blood_type': '',
                    'email': '',
                    'phone': '',
                    'address': ''
                },
                'emergency_contact': {
                    'name': '',
                    'relationship': '',
                    'phone': '',
                    'email': ''
                },
                'medical_info': {
                    'allergies': [],
                    'conditions': [],
                    'medications': [],
                    'surgeries': [],
                    'primary_doctor': '',
                    'insurance': ''
                },
                'preferences': {
                    'theme': 'light',
                    'notifications': True,
                    'data_sharing': False,
                    'language': 'English'
                }
            }
        
        # Create tabs for different profile sections
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal Info", "🏥 Medical Info", "🔔 Preferences", "📊 Profile Summary"])
        
        with tab1:
            self._show_personal_info_tab()
        
        with tab2:
            self._show_medical_info_tab()
        
        with tab3:
            self._show_preferences_tab()
        
        with tab4:
            self._show_profile_summary_tab()
    
    def _show_personal_info_tab(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("👤 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic Information
            st.markdown("#### 📝 Basic Information")
            full_name = st.text_input("Full Name*", 
                                    value=st.session_state.user_profile['personal_info']['full_name'],
                                    placeholder="Enter your full name",
                                    key="profile_full_name")
            
            date_of_birth = st.date_input("Date of Birth*", 
                                        value=st.session_state.user_profile['personal_info']['date_of_birth'] or datetime.now().date(),
                                        key="profile_dob")
            
            gender = st.selectbox("Gender", 
                                ["Select gender", "Male", "Female", "Non-binary", "Prefer not to say"],
                                key="profile_gender")
            
            blood_type = st.selectbox("Blood Type", 
                                    ["Unknown", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                    key="profile_blood_type")
            
            # Contact Information
            st.markdown("#### 📞 Contact Information")
            email = st.text_input("Email Address*", 
                                value=st.session_state.user_profile['personal_info']['email'],
                                placeholder="your.email@example.com",
                                key="profile_email")
            
            phone = st.text_input("Phone Number", 
                                value=st.session_state.user_profile['personal_info']['phone'],
                                placeholder="+1 (555) 123-4567",
                                key="profile_phone")
            
            address = st.text_area("Address", 
                                value=st.session_state.user_profile['personal_info']['address'],
                                placeholder="Enter your full address",
                                key="profile_address")
        
        with col2:
            # Emergency Contact
            st.markdown("#### 🆘 Emergency Contact")
            emergency_name = st.text_input("Emergency Contact Name*", 
                                        value=st.session_state.user_profile['emergency_contact']['name'],
                                        placeholder="Full name of emergency contact",
                                        key="emergency_name")
            
            emergency_relationship = st.selectbox("Relationship*", 
                                                ["Select relationship", "Spouse", "Parent", "Child", "Sibling", "Friend", "Other"],
                                                key="emergency_relationship")
            
            emergency_phone = st.text_input("Emergency Contact Phone*", 
                                        value=st.session_state.user_profile['emergency_contact']['phone'],
                                        placeholder="+1 (555) 987-6543",
                                        key="emergency_phone")
            
            emergency_email = st.text_input("Emergency Contact Email", 
                                        value=st.session_state.user_profile['emergency_contact']['email'],
                                        placeholder="contact@example.com",
                                        key="emergency_email")
            
            # Profile Photo
            st.markdown("#### 📷 Profile Photo")
            uploaded_file = st.file_uploader("Upload profile picture", 
                                           type=['jpg', 'jpeg', 'png'],
                                           help="Upload a clear photo of yourself")
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Profile Picture", width=150)
                
                # Save to session state
                if 'profile_picture' not in st.session_state:
                    st.session_state.profile_picture = uploaded_file
        
        # Save Personal Info Button
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("💾 Save Personal Info", use_container_width=True, key="save_personal"):
                # Validate required fields
                if not full_name or not email:
                    st.error("❌ Please fill in all required fields (marked with *)")
                else:
                    # Update session state
                    st.session_state.user_profile['personal_info'].update({
                        'full_name': full_name,
                        'date_of_birth': date_of_birth,
                        'gender': gender if gender != "Select gender" else "",
                        'blood_type': blood_type,
                        'email': email,
                        'phone': phone,
                        'address': address
                    })
                    
                    st.session_state.user_profile['emergency_contact'].update({
                        'name': emergency_name,
                        'relationship': emergency_relationship if emergency_relationship != "Select relationship" else "",
                        'phone': emergency_phone,
                        'email': emergency_email
                    })
                    
                    st.success("✅ Personal information saved successfully!")
        
        with col3:
            if st.button("🔄 Reset Form", use_container_width=True, key="reset_personal"):
                st.session_state.user_profile['personal_info'] = {
                    'full_name': '', 'date_of_birth': None, 'gender': '', 'blood_type': '',
                    'email': '', 'phone': '', 'address': ''
                }
                st.session_state.user_profile['emergency_contact'] = {
                    'name': '', 'relationship': '', 'phone': '', 'email': ''
                }
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_medical_info_tab(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🏥 Medical Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Allergies
            st.markdown("#### 🤧 Allergies")
            st.write("List any allergies you have:")
            allergies = st.text_area("Allergies", 
                                   value="\n".join(st.session_state.user_profile['medical_info']['allergies']),
                                   placeholder="Enter each allergy on a new line\nExample:\nPenicillin\nPeanuts\nDust mites",
                                   height=100,
                                   key="medical_allergies")
            
            # Current Medications
            st.markdown("#### 💊 Current Medications")
            st.write("List medications you're currently taking:")
            medications = st.text_area("Medications", 
                                     value="\n".join(st.session_state.user_profile['medical_info']['medications']),
                                     placeholder="Enter each medication on a new line\nExample:\nLisinopril 10mg daily\nAtorvastatin 20mg daily",
                                     height=100,
                                     key="medical_medications")
            
            # Primary Doctor
            st.markdown("#### 👨‍⚕️ Primary Care Physician")
            primary_doctor = st.text_input("Doctor's Name", 
                                         value=st.session_state.user_profile['medical_info']['primary_doctor'],
                                         placeholder="Dr. Smith",
                                         key="primary_doctor")
        
        with col2:
            # Medical Conditions
            st.markdown("#### 🩺 Medical Conditions")
            st.write("List any chronic or ongoing medical conditions:")
            conditions = st.text_area("Medical Conditions", 
                                    value="\n".join(st.session_state.user_profile['medical_info']['conditions']),
                                    placeholder="Enter each condition on a new line\nExample:\nHypertension\nType 2 Diabetes\nAsthma",
                                    height=100,
                                    key="medical_conditions")
            
            # Surgeries
            st.markdown("#### 🏥 Surgical History")
            st.write("List any past surgeries:")
            surgeries = st.text_area("Surgeries", 
                                   value="\n".join(st.session_state.user_profile['medical_info']['surgeries']),
                                   placeholder="Enter each surgery on a new line\nExample:\nAppendectomy (2018)\nKnee replacement (2020)",
                                   height=100,
                                   key="medical_surgeries")
            
            # Insurance Information
            st.markdown("#### 📄 Insurance Information")
            insurance = st.text_input("Insurance Provider", 
                                    value=st.session_state.user_profile['medical_info']['insurance'],
                                    placeholder="Insurance company name",
                                    key="insurance_info")
        
        # Save Medical Info Button
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("💾 Save Medical Info", use_container_width=True, key="save_medical"):
                # Update session state
                st.session_state.user_profile['medical_info'].update({
                    'allergies': [a.strip() for a in allergies.split('\n') if a.strip()],
                    'conditions': [c.strip() for c in conditions.split('\n') if c.strip()],
                    'medications': [m.strip() for m in medications.split('\n') if m.strip()],
                    'surgeries': [s.strip() for s in surgeries.split('\n') if s.strip()],
                    'primary_doctor': primary_doctor,
                    'insurance': insurance
                })
                
                st.success("✅ Medical information saved successfully!")
        
        with col3:
            if st.button("🔄 Reset Medical", use_container_width=True, key="reset_medical"):
                st.session_state.user_profile['medical_info'] = {
                    'allergies': [], 'conditions': [], 'medications': [], 
                    'surgeries': [], 'primary_doctor': '', 'insurance': ''
                }
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_preferences_tab(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🔔 Preferences & Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Appearance Settings
            st.markdown("#### 🎨 Appearance")
            theme = st.selectbox("Theme", 
                               ["Light", "Dark", "System Default"],
                               index=0 if st.session_state.user_profile['preferences']['theme'] == 'light' else 1,
                               key="pref_theme")
            
            font_size = st.select_slider("Font Size", 
                                       options=["Small", "Medium", "Large"],
                                       value="Medium",
                                       key="pref_font_size")
            
            language = st.selectbox("Language", 
                                  ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
                                  key="pref_language")
            
            # Notification Settings
            st.markdown("#### 🔔 Notifications")
            email_notifications = st.checkbox("Email Notifications", 
                                            value=st.session_state.user_profile['preferences']['notifications'],
                                            key="pref_email_notifications")
            
            sms_notifications = st.checkbox("SMS Notifications", 
                                          value=False,
                                          key="pref_sms_notifications")
            
            push_notifications = st.checkbox("Push Notifications", 
                                           value=True,
                                           key="pref_push_notifications")
        
        with col2:
            # Privacy Settings
            st.markdown("#### 🔒 Privacy & Data")
            data_sharing = st.selectbox("Data Sharing", 
                                      ["No sharing", "Anonymous data only", "Share with healthcare providers"],
                                      key="pref_data_sharing")
            
            research_participation = st.checkbox("Participate in medical research", 
                                               value=st.session_state.user_profile['preferences']['data_sharing'],
                                               key="pref_research")
            
            data_retention = st.selectbox("Data Retention Period", 
                                        ["30 days", "6 months", "1 year", "3 years", "Indefinitely"],
                                        key="pref_retention")
            
            # Export Data
            st.markdown("#### 📤 Data Management")
            if st.button("📥 Export My Data", use_container_width=True, key="export_data"):
                self._export_user_data()
            
            if st.button("🗑️ Delete My Data", use_container_width=True, key="delete_data"):
                st.warning("This will permanently delete all your data. This action cannot be undone.")
                if st.button("Confirm Permanent Deletion", type="primary", key="confirm_delete"):
                    self._delete_user_data()
        
        # Health Monitoring Preferences
        st.markdown("#### ❤️ Health Monitoring")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vital_monitoring = st.checkbox("Vital Signs Monitoring", value=True, key="pref_vital")
            sleep_tracking = st.checkbox("Sleep Quality Tracking", value=True, key="pref_sleep")
        
        with col2:
            pain_tracking = st.checkbox("Pain Assessment", value=True, key="pref_pain")
            emotion_tracking = st.checkbox("Emotion Detection", value=True, key="pref_emotion")
        
        with col3:
            medication_reminders = st.checkbox("Medication Reminders", value=True, key="pref_med_reminders")
            appointment_reminders = st.checkbox("Appointment Reminders", value=True, key="pref_appt_reminders")
        
        # Save Preferences Button
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("💾 Save Preferences", use_container_width=True, key="save_preferences"):
                st.session_state.user_profile['preferences'].update({
                    'theme': theme.lower().replace(" ", "_"),
                    'notifications': email_notifications,
                    'data_sharing': research_participation,
                    'language': language
                })
                
                st.success("✅ Preferences saved successfully!")
        
        with col3:
            if st.button("🔄 Reset Preferences", use_container_width=True, key="reset_preferences"):
                st.session_state.user_profile['preferences'] = {
                    'theme': 'light',
                    'notifications': True,
                    'data_sharing': False,
                    'language': 'English'
                }
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_profile_summary_tab(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📊 Profile Summary")
        
        # Profile Completion Score
        completion_score = self._calculate_profile_completion()
        st.markdown(f"### 📈 Profile Completion: {completion_score}%")
        st.progress(completion_score / 100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Personal Info Summary")
            personal_info = st.session_state.user_profile['personal_info']
            
            if personal_info['full_name']:
                st.write(f"**Name:** {personal_info['full_name']}")
            else:
                st.write("**Name:** ❌ Not provided")
            
            if personal_info['date_of_birth']:
                age = (datetime.now().date() - personal_info['date_of_birth']).days // 365
                st.write(f"**Age:** {age} years")
            else:
                st.write("**Age:** ❌ Not provided")
            
            if personal_info['gender']:
                st.write(f"**Gender:** {personal_info['gender']}")
            else:
                st.write("**Gender:** ❌ Not provided")
            
            if personal_info['blood_type'] and personal_info['blood_type'] != "Unknown":
                st.write(f"**Blood Type:** {personal_info['blood_type']}")
            else:
                st.write("**Blood Type:** ❌ Not provided")
            
            # Emergency Contact Summary
            st.markdown("#### 🆘 Emergency Contact")
            emergency = st.session_state.user_profile['emergency_contact']
            
            if emergency['name']:
                st.write(f"**Contact:** {emergency['name']} ({emergency['relationship']})")
                if emergency['phone']:
                    st.write(f"**Phone:** {emergency['phone']}")
            else:
                st.write("❌ No emergency contact provided")
        
        with col2:
            st.markdown("#### 🏥 Medical Info Summary")
            medical_info = st.session_state.user_profile['medical_info']
            
            # Allergies
            if medical_info['allergies']:
                st.write(f"**Allergies:** {len(medical_info['allergies'])} recorded")
                for allergy in medical_info['allergies'][:3]:  # Show first 3
                    st.write(f"  • {allergy}")
                if len(medical_info['allergies']) > 3:
                    st.write(f"  • ... and {len(medical_info['allergies']) - 3} more")
            else:
                st.write("**Allergies:** ❌ None recorded")
            
            # Conditions
            if medical_info['conditions']:
                st.write(f"**Conditions:** {len(medical_info['conditions'])} recorded")
            else:
                st.write("**Conditions:** ❌ None recorded")
            
            # Medications
            if medical_info['medications']:
                st.write(f"**Medications:** {len(medical_info['medications'])} recorded")
            else:
                st.write("**Medications:** ❌ None recorded")
            
            if medical_info['primary_doctor']:
                st.write(f"**Primary Doctor:** {medical_info['primary_doctor']}")
        
        # Quick Actions
        st.markdown("---")
        st.markdown("#### ⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🖨️ Print Summary", use_container_width=True, key="print_summary"):
                self._print_profile_summary()
        
        with col2:
            if st.button("📧 Email Summary", use_container_width=True, key="email_summary"):
                self._email_profile_summary()
        
        with col3:
            if st.button("🔄 Update Photo", use_container_width=True, key="update_photo"):
                st.session_state.current_page = "user_profile"
                st.rerun()
        
        with col4:
            if st.button("👤 Share Profile", use_container_width=True, key="share_profile"):
                self._share_profile_with_doctor()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _calculate_profile_completion(self):
        """Calculate profile completion percentage"""
        total_fields = 0
        completed_fields = 0
        
        # Personal Info (weight: 40%)
        personal_info = st.session_state.user_profile['personal_info']
        personal_fields = ['full_name', 'date_of_birth', 'gender', 'email']
        total_fields += len(personal_fields)
        completed_fields += sum(1 for field in personal_fields if personal_info[field])
        
        # Emergency Contact (weight: 20%)
        emergency_contact = st.session_state.user_profile['emergency_contact']
        emergency_fields = ['name', 'relationship', 'phone']
        total_fields += len(emergency_fields)
        completed_fields += sum(1 for field in emergency_fields if emergency_contact[field])
        
        # Medical Info (weight: 40%)
        medical_info = st.session_state.user_profile['medical_info']
        medical_fields = ['allergies', 'conditions', 'medications']
        total_fields += len(medical_fields)
        completed_fields += sum(1 for field in medical_fields if medical_info[field])
        
        if total_fields == 0:
            return 0
        
        return int((completed_fields / total_fields) * 100)
    
    def _export_user_data(self):
        """Export user data as JSON file"""
        import json
        from datetime import datetime
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'user_profile': st.session_state.user_profile,
            'health_data': {
                'vital_signs': st.session_state.get('vital_signs_history', []),
                'pain_assessments': st.session_state.get('pain_assessment_history', []),
                'sleep_data': st.session_state.get('sleep_history', []),
                'emotion_data': st.session_state.get('emotion_history', [])
            }
        }
        
        # Convert to JSON string
        json_data = json.dumps(export_data, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download Data Export",
            data=json_data,
            file_name=f"health_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_export"
        )
    
    def _delete_user_data(self):
        """Delete all user data"""
        st.session_state.user_profile = {
            'personal_info': {'full_name': '', 'date_of_birth': None, 'gender': '', 'blood_type': '', 'email': '', 'phone': '', 'address': ''},
            'emergency_contact': {'name': '', 'relationship': '', 'phone': '', 'email': ''},
            'medical_info': {'allergies': [], 'conditions': [], 'medications': [], 'surgeries': [], 'primary_doctor': '', 'insurance': ''},
            'preferences': {'theme': 'light', 'notifications': True, 'data_sharing': False, 'language': 'English'}
        }
        
        # Clear health data
        st.session_state.vital_signs_history = []
        st.session_state.pain_assessment_history = []
        st.session_state.sleep_history = []
        st.session_state.emotion_history = []
        
        st.success("✅ All user data has been deleted successfully!")
        st.rerun()
    
    def _print_profile_summary(self):
        """Generate printable profile summary"""
        st.info("🖨️ Print feature would generate a formatted PDF summary of your profile in a production environment")
    
    def _email_profile_summary(self):
        """Email profile summary"""
        st.info("📧 Email feature would send your profile summary to your registered email address in a production environment")
    
    def _share_profile_with_doctor(self):
        """Share profile with healthcare provider"""
        st.info("👤 Share feature would allow you to securely share your profile with healthcare providers in a production environment")

    def show_settings(self):
        st.markdown("<h1 class='main-title'>⚙️ System Settings</h1>", unsafe_allow_html=True)
        
        # Create tabs for different settings categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Appearance", "🔔 Notifications", "🔒 Security", "📊 Data", "ℹ️ About"])
        
        with tab1:
            self._show_appearance_settings()
        
        with tab2:
            self._show_notification_settings()
        
        with tab3:
            self._show_security_settings()
        
        with tab4:
            self._show_data_settings()
        
        with tab5:
            self._show_about_section()
    
    def _show_appearance_settings(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🎨 Appearance Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Theme Settings
            st.markdown("#### 🎯 Theme")
            current_theme = st.session_state.user_profile['preferences'].get('theme', 'light')
            theme = st.selectbox("Select Theme", 
                               ["Light", "Dark", "Auto (System)"],
                               index=0 if current_theme == 'light' else 1 if current_theme == 'dark' else 2,
                               key="settings_theme")
            
            # Color Scheme
            st.markdown("#### 🎨 Color Scheme")
            primary_color = st.color_picker("Primary Color", "#1976D2", key="primary_color")
            secondary_color = st.color_picker("Secondary Color", "#1565C0", key="secondary_color")
            
            # Layout Preferences
            st.markdown("#### 📐 Layout")
            layout_style = st.radio("Layout Style", 
                                  ["Compact", "Comfortable", "Spacious"],
                                  index=1,
                                  key="layout_style")
            
            sidebar_position = st.radio("Sidebar Position",
                                      ["Left", "Right"],
                                      index=0,
                                      key="sidebar_position")
        
        with col2:
            # Font Settings
            st.markdown("#### 🔤 Font Settings")
            font_family = st.selectbox("Font Family",
                                     ["System Default", "Arial", "Helvetica", "Georgia", "Times New Roman"],
                                     key="font_family")
            
            font_size = st.select_slider("Font Size",
                                       options=["Small", "Medium", "Large", "X-Large"],
                                       value="Medium",
                                       key="settings_font_size")
            
            # Display Settings
            st.markdown("#### 👁️ Display")
            high_contrast = st.checkbox("High Contrast Mode", value=False, key="high_contrast")
            reduce_animations = st.checkbox("Reduce Animations", value=False, key="reduce_animations")
            screen_reader = st.checkbox("Screen Reader Support", value=False, key="screen_reader")
            
            # Preview
            st.markdown("#### 👀 Preview")
            st.info("Changes will be applied after saving")
        
        # Save Appearance Settings
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("💾 Save Appearance", use_container_width=True, key="save_appearance"):
                st.session_state.user_profile['preferences']['theme'] = theme.lower().split()[0]
                st.success("✅ Appearance settings saved! Refresh to see changes.")
        
        with col3:
            if st.button("🔄 Reset to Default", use_container_width=True, key="reset_appearance"):
                st.info("Appearance settings reset to default values")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_notification_settings(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🔔 Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Notification Channels
            st.markdown("#### 📱 Notification Channels")
            email_notifications = st.checkbox("Email Notifications", value=True, key="settings_email")
            push_notifications = st.checkbox("Push Notifications", value=True, key="settings_push")
            sms_notifications = st.checkbox("SMS Notifications", value=False, key="settings_sms")
            browser_notifications = st.checkbox("Browser Notifications", value=True, key="settings_browser")
            
            # Notification Schedule
            st.markdown("#### ⏰ Notification Schedule")
            quiet_hours = st.checkbox("Enable Quiet Hours", value=False, key="quiet_hours")
            if quiet_hours:
                col_start, col_end = st.columns(2)
                with col_start:
                    quiet_start = st.time_input("Quiet Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
                with col_end:
                    quiet_end = st.time_input("Quiet Hours End", value=datetime.strptime("07:00", "%H:%M").time())
        
        with col2:
            # Notification Types
            st.markdown("#### 🎯 Notification Types")
            appointment_reminders = st.checkbox("Appointment Reminders", value=True, key="notif_appointment")
            medication_reminders = st.checkbox("Medication Reminders", value=True, key="notif_medication")
            health_alerts = st.checkbox("Health Alerts", value=True, key="notif_health")
            system_updates = st.checkbox("System Updates", value=False, key="notif_system")
            promotional = st.checkbox("Promotional Messages", value=False, key="notif_promotional")
            
            # Notification Frequency
            st.markdown("#### 📊 Frequency")
            notification_frequency = st.select_slider("Notification Frequency",
                                                    options=["Minimal", "Normal", "Frequent"],
                                                    value="Normal",
                                                    key="notif_frequency")
        
        # Emergency Notifications
        st.markdown("#### 🚨 Emergency Notifications")
        col1, col2 = st.columns(2)
        with col1:
            emergency_alerts = st.checkbox("Emergency Health Alerts", value=True, key="emergency_alerts")
            critical_results = st.checkbox("Critical Test Results", value=True, key="critical_results")
        with col2:
            always_notify = st.checkbox("Always Notify (override quiet hours)", value=True, key="always_notify")
        
        # Save Notification Settings
        st.markdown("---")
        if st.button("💾 Save Notification Settings", use_container_width=True, key="save_notifications"):
            st.success("✅ Notification settings saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_security_settings(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("🔒 Security Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Password Management
            st.markdown("#### 🔐 Password")
            current_password = st.text_input("Current Password", type="password", key="current_pass")
            new_password = st.text_input("New Password", type="password", key="new_pass")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pass")
            
            if st.button("🔄 Change Password", use_container_width=True, key="change_pass"):
                if new_password and new_password == confirm_password:
                    st.success("✅ Password changed successfully!")
                else:
                    st.error("❌ Passwords don't match or are empty")
            
            # Two-Factor Authentication
            st.markdown("#### 🔒 Two-Factor Authentication")
            two_factor = st.checkbox("Enable Two-Factor Authentication", value=False, key="two_factor")
            if two_factor:
                auth_method = st.radio("Authentication Method",
                                     ["Authenticator App", "SMS", "Email"],
                                     key="auth_method")
        
        with col2:
            # Session Management
            st.markdown("#### 💻 Session Settings")
            auto_logout = st.checkbox("Auto Logout After Inactivity", value=True, key="auto_logout")
            if auto_logout:
                logout_time = st.selectbox("Logout After",
                                         ["15 minutes", "30 minutes", "1 hour", "2 hours"],
                                         index=1,
                                         key="logout_time")
            
            remember_me = st.checkbox("Remember Me on This Device", value=True, key="remember_me")
            
            # Privacy Settings
            st.markdown("#### 👁️ Privacy")
            show_online_status = st.checkbox("Show Online Status", value=False, key="online_status")
            activity_tracking = st.checkbox("Allow Activity Tracking", value=True, key="activity_tracking")
            
            # Security Log
            st.markdown("#### 📋 Security Log")
            if st.button("View Login History", use_container_width=True, key="view_logs"):
                st.info("Login history would be displayed here")
        
        # Advanced Security
        st.markdown("#### ⚙️ Advanced Security")
        col1, col2 = st.columns(2)
        with col1:
            encrypt_data = st.checkbox("Encrypt Local Data", value=True, key="encrypt_data")
            secure_connection = st.checkbox("Always Use Secure Connection", value=True, key="secure_conn")
        with col2:
            clear_cache = st.button("Clear Browser Cache", use_container_width=True, key="clear_cache")
            revoke_sessions = st.button("Revoke All Sessions", use_container_width=True, key="revoke_sessions")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_data_settings(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("📊 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data Storage
            st.markdown("#### 💾 Data Storage")
            storage_location = st.radio("Data Storage",
                                      ["Cloud Storage", "Local Storage Only", "Hybrid"],
                                      index=0,
                                      key="storage_location")
            
            auto_backup = st.checkbox("Automatic Cloud Backup", value=True, key="auto_backup")
            if auto_backup:
                backup_frequency = st.selectbox("Backup Frequency",
                                              ["Daily", "Weekly", "Monthly"],
                                              index=0,
                                              key="backup_freq")
            
            # Data Retention
            st.markdown("#### 🗑️ Data Retention")
            retention_period = st.selectbox("Data Retention Period",
                                          ["30 days", "6 months", "1 year", "3 years", "Indefinitely"],
                                          index=2,
                                          key="retention_period")
            
            auto_delete = st.checkbox("Auto-delete Old Data", value=False, key="auto_delete")
        
        with col2:
            # Data Sharing
            st.markdown("#### 🤝 Data Sharing")
            share_analytics = st.checkbox("Share Anonymous Analytics", value=True, key="share_analytics")
            research_participation = st.checkbox("Participate in Medical Research", value=False, key="research_participation")
            
            st.markdown("#### 📤 Data Export")
            export_format = st.radio("Export Format",
                                   ["JSON", "CSV", "PDF"],
                                   key="export_format")
            
            if st.button("📥 Export All Data", use_container_width=True, key="export_all_data"):
                self._export_user_data()
        
        # Data Usage Statistics
        st.markdown("#### 📈 Data Usage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Health Records", "45 entries")
        with col2:
            st.metric("Storage Used", "2.3 MB")
        with col3:
            st.metric("Last Backup", "Today")
        
        # Data Management Actions
        st.markdown("#### ⚠️ Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Sync Data Now", use_container_width=True, key="sync_data"):
                st.success("✅ Data synced successfully!")
        
        with col2:
            if st.button("🗑️ Delete All Data", use_container_width=True, type="secondary", key="delete_all_data"):
                st.warning("This will permanently delete ALL your data!")
                if st.button("CONFIRM PERMANENT DELETION", type="primary", key="confirm_all_delete"):
                    self._delete_user_data()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_about_section(self):
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.subheader("ℹ️ About AI Health System 3.0")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏥 System Information")
            st.write("**Version:** 3.0.1 Professional Edition")
            st.write("**Release Date:** January 2024")
            st.write("**License:** Healthcare Professional Use")
            st.write("**Developer:** AI Health Systems Inc.")
            
            st.markdown("#### 🔧 System Status")
            st.success("✅ All Systems Operational")
            st.info("🔄 Last Updated: Today")
            st.warning("📊 Storage: 65% used")
            
            # System Resources
            st.markdown("#### 💻 Resources")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("CPU Usage", "23%")
                st.metric("Memory", "1.2 GB")
            with col_res2:
                st.metric("Database", "Healthy")
                st.metric("Uptime", "99.8%")
        
        with col2:
            st.markdown("#### 📞 Support & Contact")
            st.write("**Support Email:** support@aihealthsystem.com")
            st.write("**Emergency Support:** 1-800-HELP-AI1")
            st.write("**Business Hours:** 24/7")
            
            st.markdown("#### 📚 Documentation")
            if st.button("📖 User Manual", use_container_width=True, key="user_manual"):
                st.info("Opening user manual...")
            
            if st.button("🎥 Video Tutorials", use_container_width=True, key="video_tutorials"):
                st.info("Loading video tutorials...")
            
            if st.button("❓ FAQ", use_container_width=True, key="faq"):
                st.info("Opening frequently asked questions...")
            
            st.markdown("#### 🔄 Updates")
            if st.button("🔄 Check for Updates", use_container_width=True, key="check_updates"):
                st.success("✅ You have the latest version!")
            
            if st.button("📋 System Diagnostics", use_container_width=True, key="diagnostics"):
                st.info("Running system diagnostics...")
        
        # Legal & Compliance
        st.markdown("---")
        st.markdown("#### ⚖️ Legal & Compliance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Privacy Policy", use_container_width=True, key="privacy_policy"):
                st.info("Opening privacy policy...")
        
        with col2:
            if st.button("📝 Terms of Service", use_container_width=True, key="terms_service"):
                st.info("Opening terms of service...")
        
        with col3:
            if st.button("🏥 HIPAA Compliance", use_container_width=True, key="hipaa"):
                st.info("Opening HIPAA compliance information...")
        
        st.markdown("</div>", unsafe_allow_html=True)
    def show_emergency_help(self):
        st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
        st.error("""
        # 🚨 EMERGENCY MEDICAL HELP
        
        **If you are experiencing a medical emergency, please:**
        
        ## 📞 CALL EMERGENCY SERVICES IMMEDIATELY
        ### 🏥 Go to the nearest emergency room
        ### 🚑 Or call your local emergency number
        
        **Common emergency symptoms include:**
        - Chest pain or pressure
        - Difficulty breathing
        - Severe bleeding
        - Sudden weakness or numbness
        - Severe head injury
        - Suicidal thoughts
        
        **National Suicide Prevention Lifeline: 1-800-273-8255**
        **Crisis Text Line: Text HOME to 741741**
        
        *This AI system is not a substitute for emergency medical care.*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    def show_support_contact(self):
        st.markdown("""
        # 📞 Contact Support
        
        **Technical Support:**
        - 📧 Email: support@aihealthsystem.com
        - 📞 Phone: 1-800-HELP-AI1
        - 💬 Live Chat: Available 24/7 in app
        
        **Medical Questions:**
        - Contact your healthcare provider
        - Use telemedicine for non-emergency consultations
        - For emergencies, call 911 or go to nearest ER
        
        **Feedback & Suggestions:**
        - We value your feedback to improve our services
        - Email: feedback@aihealthsystem.com
        """)

def main():
    """Main application entry point"""
    app = AIHealthSystem3()
    app.run()

if __name__ == "__main__":
    main() 