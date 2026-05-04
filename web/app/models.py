from django.db import models
from django.contrib.auth.models import User

class CareerResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=100)
    salary = models.FloatField()
    cluster = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.role

from django.contrib.auth.models import User
from django.db import models