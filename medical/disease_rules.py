class DiseaseRules:
    @staticmethod
    def migraine(s):
        return 'headache' in s
    @staticmethod
    def flu(s):
        return 'fever' in s and ('muscle_pain' in s or 'fatigue' in s)
    @staticmethod
    def covid19(s):
        return 'fever' in s or 'dry_cough' in s or 'loss_of_smell' in s
    @staticmethod
    def meningitis(s):
        return 'fever' in s and 'headache' in s and 'stiff_neck' in s
    @staticmethod
    def dengue(s):
        return 'high_fever' in s and ('headache' in s or 'joint_pain' in s)
    @staticmethod
    def common_cold(s):
        return 'runny_nose' in s or 'congestion' in s or 'sore_throat' in s
    @staticmethod
    def acne(s):
        return 'pimples' in s

def create_disease_rules():
    return {
        'Migraine': DiseaseRules.migraine,
        'Flu': DiseaseRules.flu,
        'COVID-19': DiseaseRules.covid19,
        'Meningitis': DiseaseRules.meningitis,
        'Dengue': DiseaseRules.dengue,
        'Common Cold': DiseaseRules.common_cold,
        'Acne': DiseaseRules.acne
    }
